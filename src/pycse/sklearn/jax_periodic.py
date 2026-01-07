"""JAX Periodic Neural Network Regressor with LLPR Uncertainty.

A scikit-learn compatible regressor that uses Fourier features (sin/cos) to
capture periodic patterns in specified input dimensions, with Last-Layer
Prediction Rigidity (LLPR) uncertainty quantification.

Periodic Architecture:
----------------------
For each periodic feature x_i with period T_i, we expand using Fourier basis:
    [sin(2π·1·x_i/T_i), cos(2π·1·x_i/T_i),
     sin(2π·2·x_i/T_i), cos(2π·2·x_i/T_i),
     ...,
     sin(2π·n·x_i/T_i), cos(2π·n·x_i/T_i)]

where n is the number of harmonics. This allows the network to learn arbitrary
periodic functions with the specified period.

Non-periodic features are passed through unchanged.

The expanded features are then processed through a standard MLP:
    z_k+1 = φ(W_k @ z_k + b_k)  for k = 0, ..., L-1
    f(x) = W_out @ z_L + b_out

Learnable Periods:
-----------------
When `learn_period=True`, the periods are learned from data instead of being
fixed. The periods are parameterized using softplus to ensure positivity:
    T = softplus(T_raw) = log(1 + exp(T_raw))

A regularization term encourages periods to stay near their initial values:
    L_reg = period_reg * Σ (T - T_init)²

LLPR Uncertainty Quantification:
-------------------------------
Uses Last-Layer Prediction Rigidity to provide uncertainty estimates:
    σ²★ = α² f★ᵀ(F^T F + ζ²I)^{-1} f★

where f★ is the last-layer feature vector and F is the matrix of training features.

References:
-----------
[1] Rahimi, A., & Recht, B. (2007). "Random Features for Large-Scale Kernel
    Machines." NIPS.
[2] Tancik, M., et al. (2020). "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains." NeurIPS.
[3] Kristiadi, A., Hein, M., & Hennig, P. (2020). "Being Bayesian, Even Just
    a Bit, Fixes Overconfident ReLU Networks." ICML.

Example usage:
    from pycse.sklearn.jax_periodic import JAXPeriodicRegressor

    # Fixed period (default)
    model = JAXPeriodicRegressor(
        periodicity={0: 2*np.pi},  # x0 is periodic with period 2π
        n_harmonics=5,
    )

    # Learnable period
    model = JAXPeriodicRegressor(
        periodicity={0: 2*np.pi},  # Initial guess for period
        learn_period=True,         # Learn the actual period from data
        period_reg=0.1,            # Regularization toward initial period
    )
    model.fit(X_train, y_train)
    print(f"Learned period: {model.learned_periods_[0]:.4f}")

Requires: numpy, sklearn, jax, optax
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, vmap
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


def _silu(x: jnp.ndarray) -> jnp.ndarray:
    """SiLU (Swish) activation: x * sigmoid(x).

    Smooth, differentiable activation function.
    """
    return x * jax.nn.sigmoid(x)


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Softplus activation: log(1 + exp(x)).

    Smooth approximation to ReLU.
    """
    return jnp.logaddexp(x, 0.0)


def _relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU activation: max(0, x)."""
    return jnp.maximum(x, 0.0)


def _tanh(x: jnp.ndarray) -> jnp.ndarray:
    """Tanh activation."""
    return jnp.tanh(x)


def _softplus_inverse(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of softplus: log(exp(x) - 1).

    Used to initialize raw period parameters.
    """
    return jnp.log(jnp.exp(x) - 1.0 + 1e-10)


def _expand_periodic_features_vectorized(
    x: jnp.ndarray,
    periods: jnp.ndarray,
    expand_indices: jnp.ndarray,
    harmonic_nums: jnp.ndarray,
    is_sin: jnp.ndarray,
) -> jnp.ndarray:
    """Expand input with Fourier features using precomputed index structure.

    This function is designed to be JIT-compatible without Python conditionals.
    The structure arrays (expand_indices, harmonic_nums, is_sin) are precomputed
    at init time and describe how to build the output.

    Args:
        x: Input array of shape (n_features,).
        periods: Array of periods for each input feature, shape (n_features,).
        expand_indices: Which input feature each output comes from, shape (n_output,).
        harmonic_nums: Harmonic number for each output (0 for passthrough), shape (n_output,).
        is_sin: Whether each output is sin (True) or cos/passthrough (False), shape (n_output,).

    Returns:
        Expanded feature array of shape (n_output,).
    """
    # Get the input feature values and periods for each output
    input_vals = x[expand_indices]  # (n_output,)
    period_vals = periods[expand_indices]  # (n_output,)

    # Compute frequency for each output
    # For passthrough (harmonic_nums == 0), this will compute garbage but we mask it out
    freqs = 2.0 * jnp.pi * harmonic_nums / jnp.maximum(period_vals, 1e-10)  # (n_output,)
    phase = freqs * input_vals

    # Compute sin and cos
    sin_vals = jnp.sin(phase)
    cos_vals = jnp.cos(phase)

    # Select: passthrough (harmonic_nums == 0), sin, or cos
    is_passthrough = harmonic_nums == 0

    result = jnp.where(
        is_passthrough,
        input_vals,
        jnp.where(is_sin, sin_vals, cos_vals),
    )

    return result


def _build_expansion_indices(
    n_features: int,
    periodic_indices: np.ndarray,
    n_harmonics: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build index arrays for vectorized feature expansion.

    This is called once at init time to precompute the structure.

    Args:
        n_features: Number of input features.
        periodic_indices: Array of indices (-1 for non-periodic).
        n_harmonics: Number of harmonics for periodic features.

    Returns:
        Tuple of (expand_indices, harmonic_nums, is_sin) arrays.
    """
    expand_indices = []
    harmonic_nums = []
    is_sin = []

    for i in range(n_features):
        is_periodic = periodic_indices[i] >= 0

        if is_periodic:
            # Add Fourier features: sin and cos for each harmonic
            for h in range(1, n_harmonics + 1):
                # Sin term
                expand_indices.append(i)
                harmonic_nums.append(h)
                is_sin.append(True)
                # Cos term
                expand_indices.append(i)
                harmonic_nums.append(h)
                is_sin.append(False)
        else:
            # Non-periodic: pass through
            expand_indices.append(i)
            harmonic_nums.append(0)  # 0 indicates passthrough
            is_sin.append(False)

    return (
        np.array(expand_indices, dtype=np.int32),
        np.array(harmonic_nums, dtype=np.float64),
        np.array(is_sin, dtype=np.bool_),
    )


def _compute_expanded_dim(
    n_original_features: int,
    periodicity: Dict[int, float],
    n_harmonics: int,
) -> int:
    """Compute the dimension of the expanded feature space.

    Args:
        n_original_features: Number of original features.
        periodicity: Dict mapping feature indices to periods.
        n_harmonics: Number of harmonics.

    Returns:
        Dimension of expanded feature space.
    """
    n_periodic = len(periodicity)
    n_non_periodic = n_original_features - n_periodic

    # Each periodic feature expands to 2*n_harmonics (sin + cos for each harmonic)
    return n_non_periodic + n_periodic * 2 * n_harmonics


def _init_params(
    key: jax.random.PRNGKey,
    n_expanded_features: int,
    hidden_dims: Tuple[int, ...],
    periods_init: Optional[jnp.ndarray] = None,
    learn_period: bool = False,
) -> Dict[str, Any]:
    """Initialize network parameters with Xavier scaling.

    Args:
        key: JAX PRNG key.
        n_expanded_features: Number of expanded input features.
        hidden_dims: Tuple of hidden layer dimensions.
        periods_init: Initial period values (for learnable periods).
        learn_period: Whether periods are learnable.

    Returns:
        PyTree of parameters.
    """
    params = {"W": [], "b": []}
    dims = [n_expanded_features] + list(hidden_dims) + [1]

    for i in range(len(dims) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        in_dim = dims[i]
        out_dim = dims[i + 1]

        # Xavier initialization
        std = jnp.sqrt(2.0 / (in_dim + out_dim))
        W = jax.random.normal(k1, (in_dim, out_dim)) * std
        b = jnp.zeros(out_dim)

        params["W"].append(W)
        params["b"].append(b)

    # Add learnable period parameters if requested
    if learn_period and periods_init is not None:
        # Use softplus parameterization: period = softplus(period_raw)
        # Initialize so that softplus(period_raw) ≈ periods_init
        params["period_raw"] = _softplus_inverse(periods_init)

    return params


def _forward(
    params: Dict[str, Any],
    x_expanded: jnp.ndarray,
    activation: str = "silu",
    return_features: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Forward pass through the network.

    Args:
        params: Parameter PyTree.
        x_expanded: Expanded input array.
        activation: Activation function name.
        return_features: If True, also return last-layer features for LLPR.

    Returns:
        Scalar output f(x), or (output, features) if return_features=True.
    """
    if activation == "silu":
        phi = _silu
    elif activation == "softplus":
        phi = _softplus
    elif activation == "relu":
        phi = _relu
    elif activation == "tanh":
        phi = _tanh
    else:
        raise ValueError(f"Unknown activation: {activation}")

    n_layers = len(params["W"])
    z = x_expanded

    for i in range(n_layers - 1):
        z = phi(z @ params["W"][i] + params["b"][i])

    # Store last-layer features before final linear transformation
    features = z

    # Final layer (no activation)
    output = z @ params["W"][-1] + params["b"][-1]
    output = output.squeeze()

    if return_features:
        return output, features
    return output


def _forward_batch(
    params: Dict[str, Any],
    X_expanded: jnp.ndarray,
    activation: str = "silu",
    return_features: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Batched forward pass using vmap.

    Args:
        params: Parameter PyTree.
        X_expanded: Expanded input array of shape (n_samples, n_expanded_features).
        activation: Activation function name.
        return_features: If True, also return features.

    Returns:
        Output array of shape (n_samples,), or (outputs, features).
    """

    def forward_single(x):
        return _forward(params, x, activation, return_features)

    if return_features:
        outputs, features = vmap(forward_single)(X_expanded)
        return outputs, features
    return vmap(forward_single)(X_expanded)


class JAXPeriodicRegressor(BaseEstimator, RegressorMixin):
    """Periodic Neural Network Regressor with LLPR Uncertainty.

    This regressor uses Fourier features (sin/cos) to capture periodic patterns
    in specified input dimensions. It also provides uncertainty quantification
    using Last-Layer Prediction Rigidity (LLPR).

    The periodic representation allows the network to learn functions that are
    exactly periodic with the specified period(s), making it suitable for:
    - Angle-dependent properties (dihedral angles in molecules)
    - Time-series with known periodicity
    - Cyclic features (day of week, month of year)
    - Phase-dependent phenomena

    Parameters
    ----------
    hidden_dims : tuple of int, default=(32, 32)
        Dimensions of hidden layers.

    periodicity : float, list, dict, or None, default=None
        Specifies which features are periodic and their periods.
        Accepts multiple formats:
        - float: Feature 0 has this period (e.g., 2*np.pi)
        - list: Each position i has period[i], None for non-periodic
          (e.g., [2*np.pi, None, 24.0])
        - dict: Maps feature indices to periods (e.g., {0: 2*np.pi, 2: 24.0})
        - None: No features are periodic
        Unspecified features are treated as non-periodic.
        When learn_period=True, these values are used as initial guesses.

    n_harmonics : int, default=5
        Number of harmonics to use for each periodic feature.
        Higher values capture more complex periodic patterns but
        increase model capacity and potential overfitting.

    learn_period : bool, default=False
        Whether to learn the periods from data. If True, the periods
        specified in `periodicity` are used as initial guesses and
        optimized during training.

    period_reg : float, default=0.1
        Regularization strength for learned periods. Encourages periods
        to stay near their initial values. Only used when learn_period=True.
        Higher values = stronger regularization toward initial periods.

    activation : str, default="silu"
        Activation function for hidden layers.
        Options: "silu", "softplus", "relu", "tanh".

    learning_rate : float, default=5e-3
        Learning rate for Adam optimizer.

    weight_decay : float, default=0.0
        L2 regularization strength.

    epochs : int, default=50
        Number of training epochs.

    batch_size : int, default=32
        Minibatch size for training.

    standardize_X : bool, default=True
        Whether to standardize non-periodic input features.
        Periodic features are NOT standardized (they're expanded to Fourier).

    standardize_y : bool, default=True
        Whether to standardize target values.

    alpha_squared : float or 'auto', default='auto'
        LLPR calibration parameter for uncertainty scaling.
        If 'auto', calibrated on validation data.

    zeta_squared : float or 'auto', default='auto'
        LLPR regularization parameter for covariance matrix.
        If 'auto', calibrated on validation data.

    val_size : float, default=0.1
        Fraction of training data for validation (used for LLPR calibration).

    random_state : int, default=42
        Random seed for reproducibility.

    verbose : bool, default=False
        Whether to print training progress.

    Attributes
    ----------
    params_ : dict
        Fitted parameters of the network.

    periodicity_ : dict
        Validated periodicity specification (initial values).

    learned_periods_ : dict or None
        If learn_period=True, the learned period values after fitting.
        Maps feature indices to learned periods. None if learn_period=False.

    periodic_indices_ : ndarray
        Array marking which features are periodic.

    periods_ : ndarray
        Array of periods for each feature (learned or fixed).

    n_expanded_features_ : int
        Number of features after Fourier expansion.

    scaler_X_ : StandardScaler or None
        Fitted scaler for non-periodic input features.

    scaler_y_ : StandardScaler or None
        Fitted scaler for target values.

    n_features_in_ : int
        Number of input features.

    loss_history_ : list of float
        Training loss at each epoch.

    cov_matrix_ : ndarray
        LLPR covariance matrix F^T F.

    alpha_squared_ : float
        Calibrated LLPR alpha parameter.

    zeta_squared_ : float
        Calibrated LLPR zeta parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from pycse.sklearn.jax_periodic import JAXPeriodicRegressor
    >>>
    >>> # Fixed periods (default)
    >>> model = JAXPeriodicRegressor(
    ...     periodicity={0: 2*np.pi},
    ...     n_harmonics=3,
    ... )
    >>>
    >>> # Learnable periods
    >>> model = JAXPeriodicRegressor(
    ...     periodicity={0: 6.0},  # Initial guess
    ...     learn_period=True,
    ...     period_reg=0.1,
    ... )
    >>> model.fit(X, y)
    >>> print(f"Learned period: {model.learned_periods_[0]:.4f}")
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 32),
        periodicity: Optional[Dict[int, float]] = None,
        n_harmonics: int = 5,
        learn_period: bool = False,
        period_reg: float = 0.1,
        activation: str = "silu",
        learning_rate: float = 5e-3,
        weight_decay: float = 0.0,
        epochs: int = 50,
        batch_size: int = 32,
        standardize_X: bool = True,
        standardize_y: bool = True,
        alpha_squared: Union[float, str] = "auto",
        zeta_squared: Union[float, str] = "auto",
        val_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden_dims = hidden_dims
        self.periodicity = periodicity
        self.n_harmonics = n_harmonics
        self.learn_period = learn_period
        self.period_reg = period_reg
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.standardize_X = standardize_X
        self.standardize_y = standardize_y
        self.alpha_squared = alpha_squared
        self.zeta_squared = zeta_squared
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose

    def _validate_periodicity(self, n_features: int) -> None:
        """Validate and store periodicity specification.

        Args:
            n_features: Number of input features.
        """
        # Normalize input to dict format
        if self.periodicity is None:
            self.periodicity_ = {}
        elif isinstance(self.periodicity, (int, float, np.integer, np.floating)):
            # Single number: feature 0 has this period
            self.periodicity_ = {0: float(self.periodicity)}
        elif isinstance(self.periodicity, (list, tuple, np.ndarray)):
            # List: each position i has period[i], None means non-periodic
            self.periodicity_ = {
                i: float(p) for i, p in enumerate(self.periodicity) if p is not None
            }
        else:
            self.periodicity_ = dict(self.periodicity)

        # Validate indices
        for idx in self.periodicity_.keys():
            if not isinstance(idx, (int, np.integer)):
                raise ValueError(f"Periodicity index must be int, got {type(idx)}")
            if idx < 0 or idx >= n_features:
                raise ValueError(f"Periodicity index {idx} out of range [0, {n_features})")

        # Validate periods
        for idx, period in self.periodicity_.items():
            if period <= 0:
                raise ValueError(f"Period must be positive, got {period} for index {idx}")

        # Create arrays for JAX operations
        self.periodic_indices_ = jnp.array(
            [i if i in self.periodicity_ else -1 for i in range(n_features)],
            dtype=jnp.int32,
        )
        self.periods_ = jnp.array(
            [self.periodicity_.get(i, 1.0) for i in range(n_features)],
            dtype=jnp.float64,
        )

        # Store initial periods for regularization
        self.initial_periods_ = self.periods_.copy()

        # Compute expanded dimension
        self.n_expanded_features_ = _compute_expanded_dim(
            n_features, self.periodicity_, self.n_harmonics
        )

        # Build precomputed index arrays for vectorized feature expansion
        # This enables JIT-compatible expansion without Python conditionals
        expand_indices, harmonic_nums, is_sin = _build_expansion_indices(
            n_features, np.array(self.periodic_indices_), self.n_harmonics
        )
        self.expand_indices_ = jnp.array(expand_indices)
        self.harmonic_nums_ = jnp.array(harmonic_nums)
        self.is_sin_ = jnp.array(is_sin)

    def _get_periods(self, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get current periods (learned or fixed).

        Args:
            params: Parameter dict. If None, uses self.params_.

        Returns:
            Array of periods for each feature.
        """
        if params is None:
            params = self.params_

        if self.learn_period and "period_raw" in params:
            # Learned periods: apply softplus to get positive values
            learned_periods = _softplus(params["period_raw"])
            # Only use learned values for periodic features
            periods = jnp.where(self.periodic_indices_ >= 0, learned_periods, self.periods_)
            return periods
        else:
            return self.periods_

    def _preprocess_X(self, X: np.ndarray, fit: bool = False) -> jnp.ndarray:
        """Preprocess input features.

        For non-periodic features: standardize if requested.
        For periodic features: no standardization (applied via Fourier expansion).
        """
        X = np.atleast_2d(X).copy()
        n_features = X.shape[1]

        if self.standardize_X:
            # Only standardize non-periodic features
            non_periodic_mask = np.array([i not in self.periodicity_ for i in range(n_features)])

            if np.any(non_periodic_mask):
                if fit:
                    self.scaler_X_ = StandardScaler()
                    X[:, non_periodic_mask] = self.scaler_X_.fit_transform(X[:, non_periodic_mask])
                else:
                    X[:, non_periodic_mask] = self.scaler_X_.transform(X[:, non_periodic_mask])
            else:
                if fit:
                    self.scaler_X_ = None
        else:
            if fit:
                self.scaler_X_ = None

        return jnp.array(X, dtype=jnp.float64)

    def _expand_features(
        self, X: jnp.ndarray, periods: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Expand features with Fourier basis for periodic dimensions.

        Args:
            X: Input array of shape (n_samples, n_features).
            periods: Periods to use. If None, uses self.periods_.

        Returns:
            Expanded array of shape (n_samples, n_expanded_features).
        """
        if periods is None:
            periods = self.periods_

        # Use precomputed index arrays for vectorized expansion
        expand_indices = self.expand_indices_
        harmonic_nums = self.harmonic_nums_
        is_sin = self.is_sin_

        def expand_single(x):
            return _expand_periodic_features_vectorized(
                x,
                periods,
                expand_indices,
                harmonic_nums,
                is_sin,
            )

        return vmap(expand_single)(X)

    def _preprocess_y(self, y: np.ndarray, fit: bool = False) -> jnp.ndarray:
        """Preprocess target values."""
        y = np.asarray(y).ravel()
        if self.standardize_y:
            if fit:
                self.scaler_y_ = StandardScaler()
                y = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()
            else:
                y = self.scaler_y_.transform(y.reshape(-1, 1)).ravel()
        else:
            if fit:
                self.scaler_y_ = None
        return jnp.array(y, dtype=jnp.float64)

    def _postprocess_y(self, y: jnp.ndarray) -> np.ndarray:
        """Postprocess predictions back to original scale."""
        y = np.asarray(y)
        if self.standardize_y and self.scaler_y_ is not None:
            y = self.scaler_y_.inverse_transform(y.reshape(-1, 1)).ravel()
        return y

    def _compute_covariance(self, X_expanded: jnp.ndarray) -> None:
        """Compute F^T F covariance matrix from training data for LLPR."""
        _, features = _forward_batch(
            self.params_,
            X_expanded,
            self.activation,
            return_features=True,
        )

        self.n_llpr_features_ = features.shape[1]
        self.cov_matrix_ = features.T @ features

    def _compute_uncertainties_batch(
        self, features: jnp.ndarray, alpha_squared: float, zeta_squared: float
    ) -> jnp.ndarray:
        """Compute LLPR uncertainties for a batch of features.

        σ²★ = α² f★ᵀ(F^T F + ζ²I)^{-1} f★
        """
        reg_cov = self.cov_matrix_ + zeta_squared * jnp.eye(self.n_llpr_features_)
        inv_cov = jnp.linalg.inv(reg_cov)

        @jit
        def compute_single_uncertainty(f):
            return alpha_squared * f.T @ inv_cov @ f

        return vmap(compute_single_uncertainty)(features)

    def _calibrate_uncertainty(
        self, X_val: jnp.ndarray, X_val_expanded: jnp.ndarray, y_val: jnp.ndarray
    ) -> None:
        """Calibrate alpha_squared and zeta_squared on validation set."""
        # Get predictions and features
        y_pred = self._postprocess_y(_forward_batch(self.params_, X_val_expanded, self.activation))
        _, features = _forward_batch(
            self.params_,
            X_val_expanded,
            self.activation,
            return_features=True,
        )

        # Grid search
        alpha_candidates = jnp.logspace(-2, 2, 20)
        zeta_candidates = jnp.logspace(-8, 0, 20)

        best_nll = float("inf")
        best_alpha = 1.0
        best_zeta = 1e-6

        y_val_orig = self._postprocess_y(y_val)

        for alpha in alpha_candidates:
            for zeta in zeta_candidates:
                variances = self._compute_uncertainties_batch(features, alpha, zeta)
                variances = jnp.maximum(variances, 1e-10)  # Numerical stability

                # Negative log-likelihood
                nll = jnp.mean(
                    0.5 * ((y_val_orig - y_pred) ** 2 / variances + jnp.log(2 * jnp.pi * variances))
                )

                if nll < best_nll:
                    best_nll = nll
                    best_alpha = alpha
                    best_zeta = zeta

        self.alpha_squared_ = (
            float(best_alpha) if self.alpha_squared == "auto" else self.alpha_squared
        )
        self.zeta_squared_ = float(best_zeta) if self.zeta_squared == "auto" else self.zeta_squared

        if self.verbose:
            print(
                f"LLPR calibrated: α²={self.alpha_squared_:.2e}, "
                f"ζ²={self.zeta_squared_:.2e}, NLL={best_nll:.4f}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JAXPeriodicRegressor":
        """Fit the periodic network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.

        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : JAXPeriodicRegressor
            Fitted estimator.
        """
        from sklearn.model_selection import train_test_split

        # Preprocess data
        X = np.atleast_2d(X)
        self.n_features_in_ = X.shape[1]

        # Validate and store periodicity
        self._validate_periodicity(self.n_features_in_)

        X_proc = self._preprocess_X(X, fit=True)
        y_proc = self._preprocess_y(y, fit=True)

        # Split for validation (used for LLPR calibration)
        if self.val_size > 0:
            indices = np.arange(len(X_proc))
            train_idx, val_idx = train_test_split(
                indices, test_size=self.val_size, random_state=self.random_state
            )
            X_train = X_proc[train_idx]
            y_train = y_proc[train_idx]
            X_val = X_proc[val_idx]
            y_val = y_proc[val_idx]
        else:
            X_train = X_proc
            y_train = y_proc
            X_val, y_val = None, None

        n_samples = X_train.shape[0]

        # Initialize parameters
        key = jax.random.PRNGKey(self.random_state)
        key, init_key = jax.random.split(key)
        self.params_ = _init_params(
            init_key,
            self.n_expanded_features_,
            self.hidden_dims,
            periods_init=self.periods_ if self.learn_period else None,
            learn_period=self.learn_period,
        )

        # Create optimizer
        if self.weight_decay > 0:
            optimizer = optax.adamw(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optax.adam(learning_rate=self.learning_rate)

        opt_state = optimizer.init(self.params_)

        # JIT-compiled functions
        activation = self.activation
        periodic_indices = self.periodic_indices_
        initial_periods = self.initial_periods_
        period_reg = self.period_reg
        learn_period = self.learn_period

        # Precomputed index arrays for vectorized expansion
        expand_indices = self.expand_indices_
        harmonic_nums = self.harmonic_nums_
        is_sin = self.is_sin_

        def expand_with_params(X_batch, params):
            """Expand features using periods from params."""
            if learn_period and "period_raw" in params:
                periods = _softplus(params["period_raw"])
            else:
                periods = initial_periods

            def expand_single(x):
                return _expand_periodic_features_vectorized(
                    x, periods, expand_indices, harmonic_nums, is_sin
                )

            return vmap(expand_single)(X_batch)

        @jit
        def loss_fn(params, X_batch, y_batch):
            # Expand features with current periods
            X_expanded = expand_with_params(X_batch, params)
            preds = _forward_batch(params, X_expanded, activation)
            mse_loss = jnp.mean((preds - y_batch) ** 2)

            # Add period regularization if learning periods
            if learn_period and "period_raw" in params:
                learned_periods = _softplus(params["period_raw"])
                # Regularize toward initial periods (only for periodic features)
                period_loss = period_reg * jnp.sum(
                    jnp.where(
                        periodic_indices >= 0,
                        (learned_periods - initial_periods) ** 2,
                        0.0,
                    )
                )
                return mse_loss + period_loss

            return mse_loss

        @jit
        def train_step(params, opt_state, X_batch, y_batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, X_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Training loop
        self.loss_history_ = []
        params = self.params_

        for epoch in range(self.epochs):
            key, shuffle_key = jax.random.split(key)
            perm = jax.random.permutation(shuffle_key, n_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            epoch_losses = []
            n_batches = max(1, n_samples // self.batch_size)

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                params, opt_state, batch_loss = train_step(params, opt_state, X_batch, y_batch)
                epoch_losses.append(float(batch_loss))

            epoch_loss = np.mean(epoch_losses)
            self.loss_history_.append(epoch_loss)

            if self.verbose and (epoch + 1) % 100 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.6f}"
                if self.learn_period and "period_raw" in params:
                    learned_periods = _softplus(params["period_raw"])
                    period_strs = [
                        f"{i}:{float(learned_periods[i]):.4f}"
                        for i in range(len(learned_periods))
                        if self.periodic_indices_[i] >= 0
                    ]
                    msg += f", Periods: {{{', '.join(period_strs)}}}"
                print(msg)

        self.params_ = params

        # Update periods_ with learned values if applicable
        if self.learn_period and "period_raw" in self.params_:
            self.periods_ = self._get_periods()
            # Store learned periods as dict for easy access
            self.learned_periods_ = {
                i: float(self.periods_[i])
                for i in range(self.n_features_in_)
                if self.periodic_indices_[i] >= 0
            }
        else:
            self.learned_periods_ = None

        # Expand features with final periods for LLPR
        X_train_expanded = self._expand_features(X_train, self.periods_)

        # Compute LLPR covariance matrix
        self._compute_covariance(X_train_expanded)

        # Calibrate uncertainty parameters
        if X_val is not None and (self.alpha_squared == "auto" or self.zeta_squared == "auto"):
            X_val_expanded = self._expand_features(X_val, self.periods_)
            self._calibrate_uncertainty(X_val, X_val_expanded, y_val)
        else:
            self.alpha_squared_ = 1.0 if self.alpha_squared == "auto" else self.alpha_squared
            self.zeta_squared_ = 1e-6 if self.zeta_squared == "auto" else self.zeta_squared

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted periodic network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.atleast_2d(X)
        X_proc = self._preprocess_X(X, fit=False)
        X_expanded = self._expand_features(X_proc, self.periods_)

        preds = _forward_batch(self.params_, X_expanded, self.activation)

        return self._postprocess_y(preds)

    def predict_with_uncertainty(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with LLPR uncertainty estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        return_std : bool, default=True
            If True, return standard deviation; if False, return variance.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.

        uncertainty : ndarray of shape (n_samples,)
            Predicted uncertainties (std or variance).
        """
        X = np.atleast_2d(X)
        X_proc = self._preprocess_X(X, fit=False)
        X_expanded = self._expand_features(X_proc, self.periods_)

        preds, features = _forward_batch(
            self.params_,
            X_expanded,
            self.activation,
            return_features=True,
        )

        variances = self._compute_uncertainties_batch(
            features, self.alpha_squared_, self.zeta_squared_
        )

        y_pred = self._postprocess_y(preds)

        # Scale variance back to original units
        if self.standardize_y and self.scaler_y_ is not None:
            variances = variances * (self.scaler_y_.scale_**2)

        if return_std:
            return y_pred, np.array(jnp.sqrt(variances))
        return y_pred, np.array(variances)

    def get_fourier_features(self, X: np.ndarray) -> np.ndarray:
        """Get the Fourier-expanded features for the input.

        Useful for understanding the feature representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_expanded : ndarray of shape (n_samples, n_expanded_features)
            Fourier-expanded features.
        """
        X = np.atleast_2d(X)
        X_proc = self._preprocess_X(X, fit=False)
        X_expanded = self._expand_features(X_proc, self.periods_)
        return np.array(X_expanded)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score on given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R² score.
        """
        y_pred = self.predict(X)
        y = np.asarray(y).ravel()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


if __name__ == "__main__":
    # Example usage
    print("JAXPeriodicRegressor Example")
    print("=" * 50)

    # Generate periodic data with unknown period
    np.random.seed(42)
    n_samples = 200
    true_period = 5.5  # True period (unknown to the model)
    X = np.random.uniform(0, 3 * true_period, (n_samples, 1))
    y = np.sin(2 * np.pi * X[:, 0] / true_period) + 0.1 * np.random.randn(n_samples)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model with learnable period (initial guess = 6.0)
    print("\n1. Learnable period model:")
    model_learn = JAXPeriodicRegressor(
        hidden_dims=(32, 32),
        periodicity={0: 6.0},  # Initial guess (true period is 5.5)
        learn_period=True,
        period_reg=0.01,
        n_harmonics=5,
        epochs=50,
        random_state=42,
        verbose=True,
    )

    model_learn.fit(X_train, y_train)
    print(f"\nTrue period: {true_period:.4f}")
    print(f"Learned period: {model_learn.learned_periods_[0]:.4f}")
    print(f"R² score: {model_learn.score(X_test, y_test):.4f}")

    # Model with fixed period (correct)
    print("\n2. Fixed period model (correct period):")
    model_fixed = JAXPeriodicRegressor(
        hidden_dims=(32, 32),
        periodicity={0: true_period},
        learn_period=False,
        n_harmonics=5,
        epochs=50,
        random_state=42,
    )
    model_fixed.fit(X_train, y_train)
    print(f"R² score: {model_fixed.score(X_test, y_test):.4f}")

    # Model with fixed period (wrong)
    print("\n3. Fixed period model (wrong period):")
    model_wrong = JAXPeriodicRegressor(
        hidden_dims=(32, 32),
        periodicity={0: 6.0},  # Wrong period
        learn_period=False,
        n_harmonics=5,
        epochs=50,
        random_state=42,
    )
    model_wrong.fit(X_train, y_train)
    print(f"R² score: {model_wrong.score(X_test, y_test):.4f}")

    print("\nExample complete!")
