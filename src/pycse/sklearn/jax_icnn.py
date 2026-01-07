"""JAX Input Convex Neural Network (ICNN) Regressor.

A scikit-learn compatible regressor whose prediction function f(x) is guaranteed
to be convex in the inputs x, making it suitable for global optimization.

ICNN Architecture & Convexity Guarantee:
-----------------------------------------
The network computes:
    z_0 = 0  (or identity passthrough)
    z_{k+1} = phi(Wx_k @ x + Wz_k @ z_k + b_k)  for k = 0, ..., L-1
    f(x) = a^T @ z_L + c^T @ x + b

where:
    - phi is a convex, nondecreasing activation (softplus or relu)
    - Wz_k are constrained to be elementwise nonnegative
    - a is constrained to be elementwise nonnegative

Convexity in x is guaranteed because:
1. Affine functions of x are convex
2. Nonnegative weighted sums of convex functions are convex
3. Composition with convex, nondecreasing activations preserves convexity
   (if g is convex nondecreasing and h is convex, then g(h(x)) is convex)
4. Constraints are enforced by parameterization (softplus), not projection

Strong Convexity Option:
    When strong_convexity_mu > 0, we add (mu/2)*||x||^2 to the output,
    making the surrogate mu-strongly convex in x.

Example usage:
    from pycse.sklearn.jax_icnn import JAXICNNRegressor

    model = JAXICNNRegressor(
        hidden_dims=(32, 32),
        learning_rate=1e-3,
        epochs=50,
        batch_size=32,
        random_state=42,
    )
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    grad = model.predict_gradient(X_test[:5])

Requires: numpy, sklearn, jax, optax
"""

import os

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
import optax
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Softplus activation: log(1 + exp(x)).

    Convex and nondecreasing, suitable for ICNN hidden layers.
    """
    return jnp.logaddexp(x, 0.0)


def _relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU activation: max(0, x).

    Convex and nondecreasing, suitable for ICNN hidden layers.
    """
    return jnp.maximum(x, 0.0)


def _init_params(
    key: jax.random.PRNGKey,
    n_features: int,
    hidden_dims: Tuple[int, ...],
    nonneg_param: str = "softplus",
) -> Dict[str, Any]:
    """Initialize ICNN parameters with careful scaling for stable training.

    The initialization is designed to:
    1. Keep the network output near zero at initialization
    2. Ensure gradients flow properly through the nonnegative constraints
    3. Scale weights appropriately for the fan-in of each layer

    Args:
        key: JAX PRNG key.
        n_features: Number of input features.
        hidden_dims: Tuple of hidden layer dimensions.

    Returns:
        PyTree of parameters with structure:
        {
            "Wx": list of (n_features, hidden_dim) arrays,
            "Wz_raw": list of (hidden_dim, hidden_dim) arrays (raw, before softplus),
            "b": list of (hidden_dim,) bias vectors,
            "a_raw": (hidden_dim,) array (raw, before softplus),
            "c": (n_features,) array for skip connection,
            "b_out": scalar output bias,
        }
    """
    params = {"Wx": [], "Wz_raw": [], "b": []}

    for i in range(len(hidden_dims)):
        key, k1, k2, k3 = jax.random.split(key, 4)

        in_dim_x = n_features
        out_dim = hidden_dims[i]

        # Wx: input-to-hidden, no nonnegativity constraint
        # Use scaled initialization
        std_wx = jnp.sqrt(2.0 / (in_dim_x + out_dim))
        Wx = jax.random.normal(k1, (in_dim_x, out_dim)) * std_wx
        params["Wx"].append(Wx)

        # Wz_raw: hidden-to-hidden, will be passed through softplus/square
        if i == 0:
            # First layer has no z input, use dummy
            Wz_raw = jnp.zeros((1, out_dim))
        else:
            in_dim_z = hidden_dims[i - 1]
            # Initialize to very small values after transformation
            # For softplus: softplus(-5) ≈ 0.007, we want small positive weights
            # Scale by 1/sqrt(fan_in) to control variance of the sum
            if nonneg_param == "softplus":
                # Initialize so softplus(Wz_raw) ≈ 0.01 / sqrt(in_dim_z)
                # softplus(-4.6) ≈ 0.01
                target_weight = 0.01 / jnp.sqrt(in_dim_z)
                # softplus(x) ≈ target_weight => x ≈ log(exp(target_weight) - 1)
                init_val = jnp.log(jnp.exp(target_weight) - 1 + 1e-10)
                Wz_raw = init_val + jax.random.normal(k2, (in_dim_z, out_dim)) * 0.1
            else:
                # For square parameterization: init small
                target_weight = 0.01 / jnp.sqrt(in_dim_z)
                Wz_raw = jnp.sqrt(target_weight) * jax.random.normal(k2, (in_dim_z, out_dim))
        params["Wz_raw"].append(Wz_raw)

        # Bias initialization: small negative to prevent saturation
        b = jnp.zeros(out_dim) - 0.1
        params["b"].append(b)

    key, k1, k2 = jax.random.split(key, 3)

    # Final layer weights
    last_hidden = hidden_dims[-1]

    # a_raw: will be passed through softplus/square for nonnegativity
    # Initialize to give small output weights scaled by fan-in
    if nonneg_param == "softplus":
        target_a = 0.01 / jnp.sqrt(last_hidden)
        init_a = jnp.log(jnp.exp(target_a) - 1 + 1e-10)
        a_raw = init_a + jax.random.normal(k1, (last_hidden,)) * 0.1
    else:
        target_a = 0.01 / jnp.sqrt(last_hidden)
        a_raw = jnp.sqrt(target_a) * jax.random.normal(k1, (last_hidden,))
    params["a_raw"] = a_raw

    # c: skip connection from input to output (no constraint)
    # Initialize small for stable start
    c = jax.random.normal(k2, (n_features,)) * 0.01
    params["c"] = c

    # Output bias - learnable offset
    params["b_out"] = jnp.array(0.0)

    return params


def _forward(
    params: Dict[str, Any],
    x: jnp.ndarray,
    activation: str = "softplus",
    nonneg_param: str = "softplus",
) -> jnp.ndarray:
    """Forward pass through the ICNN.

    Args:
        params: Parameter PyTree from _init_params.
        x: Input array of shape (n_features,) for single sample.
        activation: Activation function ("softplus" or "relu").
        nonneg_param: Parameterization for nonnegative weights ("softplus" or "square").

    Returns:
        Scalar output f(x).
    """
    phi = _softplus if activation == "softplus" else _relu
    make_nonneg = _softplus if nonneg_param == "softplus" else (lambda w: w**2)

    n_layers = len(params["Wx"])
    z = None

    for i in range(n_layers):
        Wx = params["Wx"][i]
        Wz_raw = params["Wz_raw"][i]
        b = params["b"][i]

        # Contribution from input x
        pre_act = x @ Wx + b

        # Contribution from previous hidden state (if not first layer)
        if i > 0 and z is not None:
            Wz = make_nonneg(Wz_raw)  # Enforce nonnegativity
            pre_act = pre_act + z @ Wz

        # Apply activation
        z = phi(pre_act)

    # Final output layer
    a = make_nonneg(params["a_raw"])  # Nonnegative output weights
    c = params["c"]
    b_out = params["b_out"]

    # f(x) = a^T z + c^T x + b
    output = jnp.dot(a, z) + jnp.dot(c, x) + b_out

    return output


def _forward_batch(
    params: Dict[str, Any],
    X: jnp.ndarray,
    activation: str = "softplus",
    nonneg_param: str = "softplus",
) -> jnp.ndarray:
    """Batched forward pass using vmap.

    Args:
        params: Parameter PyTree.
        X: Input array of shape (n_samples, n_features).
        activation: Activation function.
        nonneg_param: Parameterization for nonnegative weights.

    Returns:
        Output array of shape (n_samples,).
    """

    def forward_single(x):
        return _forward(params, x, activation, nonneg_param)

    return vmap(forward_single)(X)


class JAXICNNRegressor(BaseEstimator, RegressorMixin):
    """Input Convex Neural Network (ICNN) Regressor with sklearn API.

    This regressor's prediction function f(x) is guaranteed to be convex
    in the inputs x, making it suitable for use as a surrogate model in
    global optimization problems.

    Convexity is achieved through the ICNN architecture:
    - Nonnegative hidden-to-hidden weights (Wz_k >= 0)
    - Nonnegative output weights from last hidden layer (a >= 0)
    - Convex, nondecreasing activation functions (softplus or relu)

    The network computes:
        z_{k+1} = phi(Wx_k @ x + Wz_k @ z_k + b_k)
        f(x) = a^T @ z_L + c^T @ x + b

    Parameters
    ----------
    hidden_dims : tuple of int, default=(32, 32)
        Dimensions of hidden layers.

    activation : str, default="softplus"
        Activation function. Must be convex and nondecreasing.
        Options: "softplus", "relu".

    nonneg_param : str, default="softplus"
        Parameterization for enforcing nonnegativity of Wz and a.
        Options: "softplus" (W = softplus(W_raw)), "square" (W = W_raw^2).

    learning_rate : float, default=5e-3
        Learning rate for Adam optimizer.

    weight_decay : float, default=0.0
        L2 regularization strength (weight decay).

    epochs : int, default=50
        Number of training epochs.

    batch_size : int, default=32
        Minibatch size for training.

    standardize_X : bool, default=True
        Whether to standardize input features to zero mean and unit variance.

    standardize_y : bool, default=True
        Whether to standardize target values.

    strong_convexity_mu : float, default=0.0
        If > 0, adds (mu/2)*||x||^2 to the output (in original X units),
        making the surrogate mu-strongly convex. Useful for optimization.

    random_state : int, default=42
        Random seed for reproducibility.

    verbose : bool, default=False
        Whether to print training progress.

    Attributes
    ----------
    params_ : dict
        Fitted parameters of the ICNN.

    scaler_X_ : StandardScaler or None
        Fitted scaler for input features (if standardize_X=True).

    scaler_y_ : StandardScaler or None
        Fitted scaler for target values (if standardize_y=True).

    n_features_in_ : int
        Number of input features.

    loss_history_ : list of float
        Training loss at each epoch.

    Examples
    --------
    >>> import numpy as np
    >>> from pycse.sklearn.jax_icnn import JAXICNNRegressor
    >>>
    >>> # Generate convex data
    >>> X = np.random.randn(100, 2)
    >>> y = np.sum(X**2, axis=1)  # Convex function
    >>>
    >>> model = JAXICNNRegressor(hidden_dims=(32, 32), epochs=100)
    >>> model.fit(X, y)
    >>> yhat = model.predict(X[:5])
    >>> grad = model.predict_gradient(X[:5])
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 32),
        activation: str = "softplus",
        nonneg_param: str = "softplus",
        learning_rate: float = 5e-3,
        weight_decay: float = 0.0,
        epochs: int = 50,
        batch_size: int = 32,
        standardize_X: bool = True,
        standardize_y: bool = True,
        strong_convexity_mu: float = 0.0,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.nonneg_param = nonneg_param
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.standardize_X = standardize_X
        self.standardize_y = standardize_y
        self.strong_convexity_mu = strong_convexity_mu
        self.random_state = random_state
        self.verbose = verbose

    def _preprocess_X(self, X: np.ndarray, fit: bool = False) -> jnp.ndarray:
        """Preprocess input features.

        Args:
            X: Input array of shape (n_samples, n_features).
            fit: Whether to fit the scaler (True during training).

        Returns:
            Preprocessed JAX array.
        """
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
        """Preprocess target values.

        Args:
            y: Target array of shape (n_samples,) or (n_samples, 1).
            fit: Whether to fit the scaler (True during training).

        Returns:
            Preprocessed JAX array of shape (n_samples,).
        """
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
        """Postprocess predictions back to original scale.

        Args:
            y: Predicted values in standardized scale.

        Returns:
            Numpy array in original scale.
        """
        y = np.asarray(y)
        if self.standardize_y and self.scaler_y_ is not None:
            y = self.scaler_y_.inverse_transform(y.reshape(-1, 1)).ravel()
        return y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JAXICNNRegressor":
        """Fit the ICNN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.

        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : JAXICNNRegressor
            Fitted estimator.
        """
        # Preprocess data
        X = np.atleast_2d(X)
        self.n_features_in_ = X.shape[1]

        X_train = self._preprocess_X(X, fit=True)
        y_train = self._preprocess_y(y, fit=True)

        n_samples = X_train.shape[0]

        # Initialize parameters
        key = jax.random.PRNGKey(self.random_state)
        key, init_key = jax.random.split(key)
        self.params_ = _init_params(
            init_key,
            self.n_features_in_,
            self.hidden_dims,
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

        # JIT-compiled loss function
        activation = self.activation
        nonneg_param = self.nonneg_param

        @jit
        def loss_fn(params, X_batch, y_batch):
            preds = _forward_batch(params, X_batch, activation, nonneg_param)
            return jnp.mean((preds - y_batch) ** 2)

        # JIT-compiled training step
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
            # Shuffle data
            key, shuffle_key = jax.random.split(key)
            perm = jax.random.permutation(shuffle_key, n_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            # Minibatch training
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
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted ICNN.

        The prediction f(x) is guaranteed to be convex in x.

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

        # Compute ICNN output
        preds = _forward_batch(self.params_, X_proc, self.activation, self.nonneg_param)

        # Inverse transform to original scale
        y_pred = self._postprocess_y(preds)

        # Add strong convexity term if requested
        if self.strong_convexity_mu > 0:
            # ||x||^2 in original X units
            X_orig = np.atleast_2d(X)
            y_pred = y_pred + 0.5 * self.strong_convexity_mu * np.sum(X_orig**2, axis=1)

        return y_pred

    def predict_gradient(self, X: np.ndarray) -> np.ndarray:
        """Compute gradient of the prediction with respect to inputs.

        The gradient is computed in the original X units, accounting for
        any input scaling.

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

        # We need to compute the gradient in original X space
        # If X is standardized, we need to account for the chain rule:
        # df/dx_orig = df/dx_std * dx_std/dx_orig = df/dx_std / scale

        X_proc = self._preprocess_X(X, fit=False)

        # Gradient of the forward pass w.r.t. standardized input
        activation = self.activation
        nonneg_param = self.nonneg_param
        params = self.params_

        def forward_single(x_std):
            return _forward(params, x_std, activation, nonneg_param)

        # vmap over samples
        grad_fn = vmap(grad(forward_single))
        grads_std = grad_fn(X_proc)
        grads_std = np.asarray(grads_std)

        # Scale gradient back to original units
        if self.standardize_X and self.scaler_X_ is not None:
            # df/dx_orig = df/dx_std / scale
            grads_orig = grads_std / self.scaler_X_.scale_
        else:
            grads_orig = grads_std

        # Scale by y scaler if present
        if self.standardize_y and self.scaler_y_ is not None:
            # y_orig = y_std * y_scale + y_mean
            # So df_orig/dx_orig = (dy_orig/dy_std) * (df_std/dx_orig)
            #                     = y_scale * df_std/dx_orig
            grads_orig = grads_orig * self.scaler_y_.scale_

        # Add strong convexity gradient if present
        if self.strong_convexity_mu > 0:
            # d/dx [(mu/2) ||x||^2] = mu * x
            grads_orig = grads_orig + self.strong_convexity_mu * X

        return grads_orig

    def predict_with_grad(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict values and gradients simultaneously.

        More efficient than calling predict() and predict_gradient() separately.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.

        grad : ndarray of shape (n_samples, n_features)
            Gradient df/dx for each sample.
        """
        X = np.atleast_2d(X)
        X_proc = self._preprocess_X(X, fit=False)

        activation = self.activation
        nonneg_param = self.nonneg_param
        params = self.params_

        def forward_single(x_std):
            return _forward(params, x_std, activation, nonneg_param)

        # Value and gradient computation
        def value_and_grad_single(x_std):
            val, g = jax.value_and_grad(forward_single)(x_std)
            return val, g

        batched_vag = vmap(value_and_grad_single)
        preds_std, grads_std = batched_vag(X_proc)

        # Postprocess predictions
        y_pred = self._postprocess_y(preds_std)

        # Scale gradients
        grads_orig = np.asarray(grads_std)
        if self.standardize_X and self.scaler_X_ is not None:
            grads_orig = grads_orig / self.scaler_X_.scale_
        if self.standardize_y and self.scaler_y_ is not None:
            grads_orig = grads_orig * self.scaler_y_.scale_

        # Add strong convexity terms
        if self.strong_convexity_mu > 0:
            X_orig = np.atleast_2d(X)
            y_pred = y_pred + 0.5 * self.strong_convexity_mu * np.sum(X_orig**2, axis=1)
            grads_orig = grads_orig + self.strong_convexity_mu * X_orig

        return y_pred, grads_orig

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

    print("JAXICNNRegressor Example")
    print("=" * 50)

    # Generate data from a convex function
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 3) * 2

    # Convex target: sum of squared features
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(n_samples)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model
    model = JAXICNNRegressor(
        hidden_dims=(32, 32),
        learning_rate=1e-3,
        epochs=50,
        batch_size=32,
        random_state=42,
        verbose=True,
    )

    print("\nFitting model...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Compute R² score
    r2 = model.score(X_test, y_test)
    print(f"\nR² score: {r2:.4f}")

    # Compute gradients
    grad = model.predict_gradient(X_test[:5])
    print(f"\nGradient shape: {grad.shape}")
    print(f"Gradient for first sample:\n{grad[0]}")

    # Test with strong convexity
    print("\n" + "=" * 50)
    print("Testing strong convexity (mu=0.1)")

    model_sc = JAXICNNRegressor(
        hidden_dims=(32, 32),
        epochs=50,
        strong_convexity_mu=0.1,
        random_state=42,
    )
    model_sc.fit(X_train, y_train)

    # The gradient should now include mu*x term
    grad_sc = model_sc.predict_gradient(X_test[:5])
    print(f"Gradient with strong convexity for first sample:\n{grad_sc[0]}")

    # Verify gradient includes mu*x term (approximately)
    grad_diff = grad_sc[0] - grad[0]
    expected_diff = 0.1 * X_test[0]
    print(f"Expected additional gradient (mu*x): {expected_diff}")
    print(f"Actual additional gradient: {grad_diff}")

    # Test predict_with_grad
    y_wg, g_wg = model.predict_with_grad(X_test[:3])
    print(f"\npredict_with_grad output shapes: y={y_wg.shape}, grad={g_wg.shape}")

    print("\nExample complete!")
