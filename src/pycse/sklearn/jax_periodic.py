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

    # Single periodic feature
    model = JAXPeriodicRegressor(
        hidden_dims=(32, 32),
        periodicity={0: 2*np.pi},  # x0 is periodic with period 2π
        n_harmonics=5,
        epochs=500,
    )
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    yhat, std = model.predict_with_uncertainty(X_test)

    # Multiple periodic features with different periods
    model = JAXPeriodicRegressor(
        periodicity={0: 2*np.pi, 2: 1.0},  # x0 period 2π, x2 period 1.0
    )

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


def _expand_periodic_features(
    x: jnp.ndarray,
    n_original_features: int,
    periodic_indices: jnp.ndarray,
    periods: jnp.ndarray,
    n_harmonics: int,
) -> jnp.ndarray:
    """Expand input with Fourier features for periodic dimensions.

    Args:
        x: Input array of shape (n_original_features,).
        n_original_features: Number of original input features.
        periodic_indices: Array of indices that are periodic (-1 for non-periodic).
        periods: Array of periods for periodic features.
        n_harmonics: Number of harmonics to use.

    Returns:
        Expanded feature array.
    """
    expanded = []

    for i in range(n_original_features):
        # Check if this feature is periodic
        is_periodic = periodic_indices[i] >= 0

        if is_periodic:
            period = periods[i]
            # Add Fourier features: sin and cos for each harmonic
            for h in range(1, n_harmonics + 1):
                freq = 2.0 * jnp.pi * h / period
                expanded.append(jnp.sin(freq * x[i]))
                expanded.append(jnp.cos(freq * x[i]))
        else:
            # Non-periodic: pass through unchanged
            expanded.append(x[i])

    return jnp.array(expanded)


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
) -> Dict[str, Any]:
    """Initialize network parameters with Xavier scaling.

    Args:
        key: JAX PRNG key.
        n_expanded_features: Number of expanded input features.
        hidden_dims: Tuple of hidden layer dimensions.

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

    periodicity : dict or None, default=None
        Dictionary mapping feature indices to their periods.
        Example: {0: 2*np.pi, 2: 1.0} means feature 0 has period 2π
        and feature 2 has period 1.0. Unspecified features are non-periodic.
        If None, no features are treated as periodic.

    n_harmonics : int, default=5
        Number of harmonics to use for each periodic feature.
        Higher values capture more complex periodic patterns but
        increase model capacity and potential overfitting.

    activation : str, default="silu"
        Activation function for hidden layers.
        Options: "silu", "softplus", "relu", "tanh".

    learning_rate : float, default=5e-3
        Learning rate for Adam optimizer.

    weight_decay : float, default=0.0
        L2 regularization strength.

    epochs : int, default=500
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
        Validated periodicity specification.

    periodic_indices_ : ndarray
        Array marking which features are periodic.

    periods_ : ndarray
        Array of periods for each feature.

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
    >>> # Generate periodic data
    >>> X = np.random.uniform(0, 2*np.pi, (100, 2))
    >>> y = np.sin(X[:, 0]) + 0.5*np.cos(2*X[:, 1])  # Periodic in both
    >>>
    >>> model = JAXPeriodicRegressor(
    ...     periodicity={0: 2*np.pi, 1: np.pi},  # Different periods
    ...     n_harmonics=3,
    ...     epochs=100
    ... )
    >>> model.fit(X, y)
    >>> yhat = model.predict(X[:5])
    >>> yhat, std = model.predict_with_uncertainty(X[:5])
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 32),
        periodicity: Optional[Dict[int, float]] = None,
        n_harmonics: int = 5,
        activation: str = "silu",
        learning_rate: float = 5e-3,
        weight_decay: float = 0.0,
        epochs: int = 500,
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
        if self.periodicity is None:
            self.periodicity_ = {}
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

        # Compute expanded dimension
        self.n_expanded_features_ = _compute_expanded_dim(
            n_features, self.periodicity_, self.n_harmonics
        )

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

    def _expand_features(self, X: jnp.ndarray) -> jnp.ndarray:
        """Expand features with Fourier basis for periodic dimensions.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            Expanded array of shape (n_samples, n_expanded_features).
        """
        n_original = X.shape[1]

        def expand_single(x):
            return _expand_periodic_features(
                x,
                n_original,
                self.periodic_indices_,
                self.periods_,
                self.n_harmonics,
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

        # Expand features
        X_expanded = self._expand_features(X_proc)

        # Split for validation (used for LLPR calibration)
        if self.val_size > 0:
            indices = np.arange(len(X_proc))
            train_idx, val_idx = train_test_split(
                indices, test_size=self.val_size, random_state=self.random_state
            )
            X_train_expanded = X_expanded[train_idx]
            y_train = y_proc[train_idx]
            X_val = X_proc[val_idx]
            X_val_expanded = X_expanded[val_idx]
            y_val = y_proc[val_idx]
        else:
            X_train_expanded = X_expanded
            y_train = y_proc
            X_val, X_val_expanded, y_val = None, None, None

        n_samples = X_train_expanded.shape[0]

        # Initialize parameters
        key = jax.random.PRNGKey(self.random_state)
        key, init_key = jax.random.split(key)
        self.params_ = _init_params(
            init_key,
            self.n_expanded_features_,
            self.hidden_dims,
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

        @jit
        def loss_fn(params, X_batch, y_batch):
            preds = _forward_batch(params, X_batch, activation)
            return jnp.mean((preds - y_batch) ** 2)

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
            X_shuffled = X_train_expanded[perm]
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
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.6f}")

        self.params_ = params

        # Compute LLPR covariance matrix
        self._compute_covariance(X_train_expanded)

        # Calibrate uncertainty parameters
        if X_val is not None and (self.alpha_squared == "auto" or self.zeta_squared == "auto"):
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
        X_expanded = self._expand_features(X_proc)

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
        X_expanded = self._expand_features(X_proc)

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
        X_expanded = self._expand_features(X_proc)
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

    # Generate periodic data
    np.random.seed(42)
    n_samples = 200
    X = np.random.uniform(0, 2 * np.pi, (n_samples, 2))

    # Target: periodic in x0 (period 2π), linear in x1
    y = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 0]) + 0.3 * X[:, 1]
    y += 0.1 * np.random.randn(n_samples)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model with periodic features
    model = JAXPeriodicRegressor(
        hidden_dims=(32, 32),
        periodicity={0: 2 * np.pi},  # x0 is periodic with period 2π
        n_harmonics=5,
        learning_rate=1e-3,
        epochs=500,
        batch_size=32,
        random_state=42,
        verbose=True,
    )

    print("\nFitting model...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    print(f"\nR² score: {r2:.4f}")

    # Predict with uncertainty
    y_pred, y_std = model.predict_with_uncertainty(X_test)
    print(f"Mean uncertainty: {np.mean(y_std):.4f}")

    # Verify periodicity
    print("\nVerifying periodicity:")
    x_test_point = np.array([[0.5, 1.0]])
    x_test_shifted = np.array([[0.5 + 2 * np.pi, 1.0]])  # Shift by period

    y1 = model.predict(x_test_point)[0]
    y2 = model.predict(x_test_shifted)[0]
    print(f"  f(0.5, 1.0) = {y1:.4f}")
    print(f"  f(0.5 + 2π, 1.0) = {y2:.4f}")
    print(f"  Difference: {abs(y1 - y2):.6f} (should be ~0)")

    # Show expanded features
    print(f"\nNumber of expanded features: {model.n_expanded_features_}")
    print(f"  Original: {model.n_features_in_}")
    print(f"  Periodic feature 0: 2 * {model.n_harmonics} = {2 * model.n_harmonics} features")
    print("  Non-periodic feature 1: 1 feature")

    print("\nExample complete!")
