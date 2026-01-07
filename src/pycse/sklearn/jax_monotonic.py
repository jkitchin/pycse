"""JAX Monotonic Neural Network Regressor with LLPR Uncertainty.

A scikit-learn compatible regressor whose prediction function f(x) is guaranteed
to be monotonic (increasing or decreasing) in specified input features, with
Last-Layer Prediction Rigidity (LLPR) uncertainty quantification.

Monotonic Architecture & Guarantee:
-----------------------------------
The network computes:
    z_0 = 0
    z_{k+1} = phi(Wx_k @ x_signed + Wz_k @ z_k + b_k)  for k = 0, ..., L-1
    f(x) = a^T @ z_L + c^T @ x_signed + b

where:
    - phi is a nondecreasing activation (softplus or relu)
    - x_signed = x * sign(monotonicity), where sign is +1 for increasing,
      -1 for decreasing, and x is unchanged for unconstrained features
    - Wx_k are constrained to be elementwise nonnegative for monotonic features
    - Wz_k are constrained to be elementwise nonnegative
    - a is constrained to be elementwise nonnegative
    - c is constrained to be nonnegative for monotonic features

Monotonicity is guaranteed because:
1. Nondecreasing activations preserve monotonicity
2. Nonnegative weights with nondecreasing activations = monotonically increasing
3. Sign flipping of inputs converts decreasing to increasing constraint
4. Unconstrained features have unconstrained weights (no monotonicity guarantee)

LLPR Uncertainty Quantification:
-------------------------------
Uses Last-Layer Prediction Rigidity to provide uncertainty estimates:
    σ²★ = α² f★ᵀ(F^T F + ζ²I)^{-1} f★

where f★ is the last-layer feature vector and F is the matrix of training features.

Example usage:
    from pycse.sklearn.jax_monotonic import JAXMonotonicRegressor

    # All features monotonically increasing
    model = JAXMonotonicRegressor(
        hidden_dims=(32, 32),
        monotonicity=1,  # +1 = increasing, -1 = decreasing, 0 = none
        epochs=500,
    )
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    yhat, std = model.predict_with_uncertainty(X_test)

    # Per-feature monotonicity
    model = JAXMonotonicRegressor(
        monotonicity=[1, -1, 0],  # x0 increasing, x1 decreasing, x2 unconstrained
    )

Requires: numpy, sklearn, jax, optax
"""

import os
from typing import Any, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, jit, vmap
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Softplus activation: log(1 + exp(x)).

    Nondecreasing, suitable for monotonic hidden layers.
    """
    return jnp.logaddexp(x, 0.0)


def _relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU activation: max(0, x).

    Nondecreasing, suitable for monotonic hidden layers.
    """
    return jnp.maximum(x, 0.0)


def _init_params(
    key: jax.random.PRNGKey,
    n_features: int,
    hidden_dims: Tuple[int, ...],
    monotonicity: jnp.ndarray,
    nonneg_param: str = "softplus",
) -> Dict[str, Any]:
    """Initialize monotonic network parameters with careful scaling.

    Args:
        key: JAX PRNG key.
        n_features: Number of input features.
        hidden_dims: Tuple of hidden layer dimensions.
        monotonicity: Array of shape (n_features,) with +1, -1, or 0.
        nonneg_param: Parameterization for nonnegative weights.

    Returns:
        PyTree of parameters.
    """
    params = {"Wx_raw": [], "Wz_raw": [], "b": []}

    for i in range(len(hidden_dims)):
        key, k1, k2, k3 = jax.random.split(key, 4)

        in_dim_x = n_features
        out_dim = hidden_dims[i]

        # Wx_raw: input-to-hidden weights
        # For monotonic features, will be passed through softplus/square
        # For unconstrained features, used directly
        if nonneg_param == "softplus":
            # Initialize so softplus gives reasonable values
            target_weight = 0.1 / jnp.sqrt(in_dim_x)
            init_val = jnp.log(jnp.exp(target_weight) - 1 + 1e-10)
            Wx_raw = init_val + jax.random.normal(k1, (in_dim_x, out_dim)) * 0.1
        else:
            target_weight = 0.1 / jnp.sqrt(in_dim_x)
            Wx_raw = jnp.sqrt(target_weight) * jax.random.normal(k1, (in_dim_x, out_dim))

        params["Wx_raw"].append(Wx_raw)

        # Wz_raw: hidden-to-hidden, will be passed through softplus/square
        if i == 0:
            Wz_raw = jnp.zeros((1, out_dim))
        else:
            in_dim_z = hidden_dims[i - 1]
            if nonneg_param == "softplus":
                target_weight = 0.01 / jnp.sqrt(in_dim_z)
                init_val = jnp.log(jnp.exp(target_weight) - 1 + 1e-10)
                Wz_raw = init_val + jax.random.normal(k2, (in_dim_z, out_dim)) * 0.1
            else:
                target_weight = 0.01 / jnp.sqrt(in_dim_z)
                Wz_raw = jnp.sqrt(target_weight) * jax.random.normal(k2, (in_dim_z, out_dim))
        params["Wz_raw"].append(Wz_raw)

        # Bias initialization
        b = jnp.zeros(out_dim) - 0.1
        params["b"].append(b)

    key, k1, k2 = jax.random.split(key, 3)

    # Final layer weights
    last_hidden = hidden_dims[-1]

    # a_raw: output weights, constrained nonnegative
    if nonneg_param == "softplus":
        target_a = 0.01 / jnp.sqrt(last_hidden)
        init_a = jnp.log(jnp.exp(target_a) - 1 + 1e-10)
        a_raw = init_a + jax.random.normal(k1, (last_hidden,)) * 0.1
    else:
        target_a = 0.01 / jnp.sqrt(last_hidden)
        a_raw = jnp.sqrt(target_a) * jax.random.normal(k1, (last_hidden,))
    params["a_raw"] = a_raw

    # c_raw: skip connection from input to output
    # Nonnegative for monotonic features, unconstrained for others
    if nonneg_param == "softplus":
        c_raw = jax.random.normal(k2, (n_features,)) * 0.01
    else:
        c_raw = jax.random.normal(k2, (n_features,)) * 0.01
    params["c_raw"] = c_raw

    # Output bias
    params["b_out"] = jnp.array(0.0)

    return params


def _forward(
    params: Dict[str, Any],
    x: jnp.ndarray,
    monotonicity: jnp.ndarray,
    activation: str = "softplus",
    nonneg_param: str = "softplus",
    return_features: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Forward pass through the monotonic network.

    Args:
        params: Parameter PyTree.
        x: Input array of shape (n_features,).
        monotonicity: Array with +1 (increasing), -1 (decreasing), 0 (none).
        activation: Activation function.
        nonneg_param: Parameterization for nonnegative weights.
        return_features: If True, also return last-layer features for LLPR.

    Returns:
        Scalar output f(x), or (output, features) if return_features=True.
    """
    phi = _softplus if activation == "softplus" else _relu
    make_nonneg = _softplus if nonneg_param == "softplus" else (lambda w: w**2)

    # Create mask for features with monotonicity constraint
    monotonic_mask = monotonicity != 0

    n_layers = len(params["Wx_raw"])
    z = None

    for i in range(n_layers):
        Wx_raw = params["Wx_raw"][i]
        Wz_raw = params["Wz_raw"][i]
        b = params["b"][i]

        # Apply nonnegative constraint to Wx for monotonic features
        # For monotonic features: use nonnegative weights
        # For unconstrained features: use raw weights
        Wx_pos = make_nonneg(Wx_raw)  # Nonnegative version
        Wx = jnp.where(monotonic_mask[:, None], Wx_pos, Wx_raw)

        # Apply sign flip to inputs for decreasing monotonicity
        # x_effective = x * sign, where sign is +1 (inc), -1 (dec), or 1 (none)
        x_effective = x * jnp.where(monotonicity == 0, 1.0, monotonicity)

        # Contribution from input
        pre_act = x_effective @ Wx + b

        # Contribution from previous hidden state
        if i > 0 and z is not None:
            Wz = make_nonneg(Wz_raw)
            pre_act = pre_act + z @ Wz

        # Apply activation
        z = phi(pre_act)

    # Store last-layer features before output
    features = z

    # Final output layer
    a = make_nonneg(params["a_raw"])
    c_raw = params["c_raw"]
    b_out = params["b_out"]

    # Skip connection: nonnegative for monotonic features
    c_pos = make_nonneg(c_raw)
    c = jnp.where(monotonic_mask, c_pos, c_raw)

    # Apply sign flip to skip connection inputs
    x_skip = x * jnp.where(monotonicity == 0, 1.0, monotonicity)

    # f(x) = a^T z + c^T x_skip + b
    output = jnp.dot(a, z) + jnp.dot(c, x_skip) + b_out

    if return_features:
        return output, features
    return output


def _forward_batch(
    params: Dict[str, Any],
    X: jnp.ndarray,
    monotonicity: jnp.ndarray,
    activation: str = "softplus",
    nonneg_param: str = "softplus",
    return_features: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Batched forward pass using vmap.

    Args:
        params: Parameter PyTree.
        X: Input array of shape (n_samples, n_features).
        monotonicity: Array of monotonicity directions.
        activation: Activation function.
        nonneg_param: Parameterization for nonnegative weights.
        return_features: If True, also return features.

    Returns:
        Output array of shape (n_samples,), or (outputs, features).
    """

    def forward_single(x):
        return _forward(params, x, monotonicity, activation, nonneg_param, return_features)

    if return_features:
        outputs, features = vmap(forward_single)(X)
        return outputs, features
    return vmap(forward_single)(X)


class JAXMonotonicRegressor(BaseEstimator, RegressorMixin):
    """Monotonic Neural Network Regressor with LLPR Uncertainty.

    This regressor's prediction function f(x) is guaranteed to be monotonic
    (increasing or decreasing) in specified input features. It also provides
    uncertainty quantification using Last-Layer Prediction Rigidity (LLPR).

    Monotonicity is achieved through:
    - Nonnegative input weights for monotonic features (after sign flip)
    - Nonnegative hidden-to-hidden weights
    - Nonnegative output weights
    - Nondecreasing activation functions

    Parameters
    ----------
    hidden_dims : tuple of int, default=(32, 32)
        Dimensions of hidden layers.

    monotonicity : int, list, or array-like, default=1
        Monotonicity constraint for each input feature:
        - +1: monotonically increasing
        - -1: monotonically decreasing
        - 0: no constraint (unconstrained)
        If a single int, applies to all features.
        If a list/array, must match the number of features.

    activation : str, default="softplus"
        Activation function. Must be nondecreasing.
        Options: "softplus", "relu".

    nonneg_param : str, default="softplus"
        Parameterization for enforcing nonnegativity.
        Options: "softplus" (W = softplus(W_raw)), "square" (W = W_raw^2).

    learning_rate : float, default=5e-3
        Learning rate for Adam optimizer.

    weight_decay : float, default=0.0
        L2 regularization strength.

    epochs : int, default=500
        Number of training epochs.

    batch_size : int, default=32
        Minibatch size for training.

    standardize_X : bool, default=True
        Whether to standardize input features.
        Note: Standardization is applied before monotonicity constraints.

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

    monotonicity_ : ndarray
        Monotonicity constraints as array (after expansion).

    scaler_X_ : StandardScaler or None
        Fitted scaler for input features.

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
    >>> from pycse.sklearn.jax_monotonic import JAXMonotonicRegressor
    >>>
    >>> # Generate monotonic data
    >>> X = np.random.randn(100, 2)
    >>> y = 2 * X[:, 0] - X[:, 1]  # Increasing in x0, decreasing in x1
    >>>
    >>> model = JAXMonotonicRegressor(
    ...     monotonicity=[1, -1],  # x0 increasing, x1 decreasing
    ...     epochs=100
    ... )
    >>> model.fit(X, y)
    >>> yhat = model.predict(X[:5])
    >>> yhat, std = model.predict_with_uncertainty(X[:5])
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 32),
        monotonicity: Union[int, list, np.ndarray] = 1,
        activation: str = "softplus",
        nonneg_param: str = "softplus",
        learning_rate: float = 5e-3,
        weight_decay: float = 0.0,
        epochs: int = 200,
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
        self.monotonicity = monotonicity
        self.activation = activation
        self.nonneg_param = nonneg_param
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

    def _validate_monotonicity(self, n_features: int) -> jnp.ndarray:
        """Validate and expand monotonicity parameter.

        Args:
            n_features: Number of input features.

        Returns:
            Array of shape (n_features,) with values in {-1, 0, +1}.
        """
        mono = self.monotonicity

        if isinstance(mono, (int, float)):
            # Scalar: apply to all features
            mono = np.full(n_features, int(mono))
        else:
            mono = np.asarray(mono)

        if mono.shape[0] != n_features:
            raise ValueError(
                f"monotonicity has {mono.shape[0]} elements but X has {n_features} features"
            )

        # Validate values
        valid_values = {-1, 0, 1}
        if not all(int(m) in valid_values for m in mono):
            raise ValueError("monotonicity values must be -1, 0, or 1")

        return jnp.array(mono, dtype=jnp.float64)

    def _preprocess_X(self, X: np.ndarray, fit: bool = False) -> jnp.ndarray:
        """Preprocess input features."""
        X = np.atleast_2d(X)
        if self.standardize_X:
            if fit:
                self.scaler_X_ = StandardScaler()
                X = self.scaler_X_.fit_transform(X)
            else:
                X = self.scaler_X_.transform(X)
        else:
            if fit:
                self.scaler_X_ = None
        return jnp.array(X, dtype=jnp.float64)

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

    def _compute_covariance(self, X: jnp.ndarray) -> None:
        """Compute F^T F covariance matrix from training data for LLPR."""
        _, features = _forward_batch(
            self.params_,
            X,
            self.monotonicity_,
            self.activation,
            self.nonneg_param,
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

    def _calibrate_uncertainty(self, X_val: jnp.ndarray, y_val: jnp.ndarray) -> None:
        """Calibrate alpha_squared and zeta_squared on validation set."""
        # Get predictions and features
        y_pred = self._postprocess_y(
            _forward_batch(
                self.params_,
                X_val,
                self.monotonicity_,
                self.activation,
                self.nonneg_param,
            )
        )
        _, features = _forward_batch(
            self.params_,
            X_val,
            self.monotonicity_,
            self.activation,
            self.nonneg_param,
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JAXMonotonicRegressor":
        """Fit the monotonic network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.

        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : JAXMonotonicRegressor
            Fitted estimator.
        """
        from sklearn.model_selection import train_test_split

        # Preprocess data
        X = np.atleast_2d(X)
        self.n_features_in_ = X.shape[1]

        # Validate and store monotonicity
        self.monotonicity_ = self._validate_monotonicity(self.n_features_in_)

        X_proc = self._preprocess_X(X, fit=True)
        y_proc = self._preprocess_y(y, fit=True)

        # Split for validation (used for LLPR calibration)
        if self.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                np.array(X_proc),
                np.array(y_proc),
                test_size=self.val_size,
                random_state=self.random_state,
            )
            X_train = jnp.array(X_train)
            y_train = jnp.array(y_train)
            X_val = jnp.array(X_val)
            y_val = jnp.array(y_val)
        else:
            X_train, y_train = X_proc, y_proc
            X_val, y_val = None, None

        n_samples = X_train.shape[0]

        # Initialize parameters
        key = jax.random.PRNGKey(self.random_state)
        key, init_key = jax.random.split(key)
        self.params_ = _init_params(
            init_key,
            self.n_features_in_,
            self.hidden_dims,
            self.monotonicity_,
            self.nonneg_param,
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
        nonneg_param = self.nonneg_param
        monotonicity = self.monotonicity_

        @jit
        def loss_fn(params, X_batch, y_batch):
            preds = _forward_batch(params, X_batch, monotonicity, activation, nonneg_param)
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
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.6f}")

        self.params_ = params

        # Compute LLPR covariance matrix
        self._compute_covariance(X_train)

        # Calibrate uncertainty parameters
        if X_val is not None and (self.alpha_squared == "auto" or self.zeta_squared == "auto"):
            self._calibrate_uncertainty(X_val, y_val)
        else:
            self.alpha_squared_ = 1.0 if self.alpha_squared == "auto" else self.alpha_squared
            self.zeta_squared_ = 1e-6 if self.zeta_squared == "auto" else self.zeta_squared

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted monotonic network.

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

        preds = _forward_batch(
            self.params_,
            X_proc,
            self.monotonicity_,
            self.activation,
            self.nonneg_param,
        )

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

        preds, features = _forward_batch(
            self.params_,
            X_proc,
            self.monotonicity_,
            self.activation,
            self.nonneg_param,
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

    def predict_gradient(self, X: np.ndarray) -> np.ndarray:
        """Compute gradient of predictions with respect to inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        grad : ndarray of shape (n_samples, n_features)
            Gradient df/dx for each sample.
        """
        X = np.atleast_2d(X)
        X_proc = self._preprocess_X(X, fit=False)

        activation = self.activation
        nonneg_param = self.nonneg_param
        monotonicity = self.monotonicity_
        params = self.params_

        def forward_single(x_std):
            return _forward(params, x_std, monotonicity, activation, nonneg_param)

        grad_fn = vmap(grad(forward_single))
        grads_std = grad_fn(X_proc)
        grads_std = np.asarray(grads_std)

        # Scale gradient back to original units
        if self.standardize_X and self.scaler_X_ is not None:
            grads_orig = grads_std / self.scaler_X_.scale_
        else:
            grads_orig = grads_std

        if self.standardize_y and self.scaler_y_ is not None:
            grads_orig = grads_orig * self.scaler_y_.scale_

        return grads_orig

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
    import numpy as np
    from sklearn.model_selection import train_test_split

    print("JAXMonotonicRegressor Example")
    print("=" * 50)

    # Generate monotonic data
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 3) * 2

    # Target: increasing in x0, decreasing in x1, unconstrained in x2
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.1 * np.random.randn(n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model with per-feature monotonicity
    model = JAXMonotonicRegressor(
        hidden_dims=(32, 32),
        monotonicity=[1, -1, 0],  # x0 increasing, x1 decreasing, x2 unconstrained
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

    # Verify monotonicity via gradients
    grads = model.predict_gradient(X_test[:10])
    print("\nGradient signs (should be +, -, any):")
    print(f"  x0 (increasing): all >= 0? {np.all(grads[:, 0] >= -1e-6)}")
    print(f"  x1 (decreasing): all <= 0? {np.all(grads[:, 1] <= 1e-6)}")
    print(f"  x2 (unconstrained): mixed signs? {grads[:, 2].min():.3f} to {grads[:, 2].max():.3f}")

    # Test with all increasing
    print("\n" + "=" * 50)
    print("Testing uniform monotonicity (all increasing)")

    X_1d = np.random.randn(150, 2)
    y_1d = X_1d[:, 0] ** 3 + 2 * X_1d[:, 1]

    model_inc = JAXMonotonicRegressor(
        monotonicity=1,  # All features increasing
        epochs=300,
        verbose=False,
    )
    model_inc.fit(X_1d, y_1d)
    print(f"R² score: {model_inc.score(X_1d, y_1d):.4f}")

    grads = model_inc.predict_gradient(X_1d[:5])
    print(f"All gradients >= 0? {np.all(grads >= -1e-6)}")

    print("\nExample complete!")
