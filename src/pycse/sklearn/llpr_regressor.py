"""
Last-Layer Prediction Rigidity (LLPR) Regressor

Implementation of the prediction rigidity formalism from:
Bigi, F., Chong, S., Ceriotti, M., & Grasselli, F. (2024).
A prediction rigidity formalism for low-cost uncertainties in trained neural networks.
Machine Learning: Science and Technology, 5, 045018.

This module provides an sklearn-compatible implementation using Flax/JAX.
"""

import warnings

import jax
import jax.numpy as jnp
from jax import random, jit
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Callable
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, train_test_split
import numpy as np

from pycse.sklearn.mip import Affine, MIPExportMixin, MIPNetwork, ReLU, StdSpec, llpr_std


class MLP(nn.Module):
    """Multi-layer perceptron with separate last layer for feature extraction."""

    features: Tuple[int, ...]
    activation: Callable = nn.silu

    def setup(self):
        # All layers except the last one
        self.hidden_layers = [nn.Dense(feat) for feat in self.features[:-1]]
        # Last layer separate for feature extraction
        self.output_layer = nn.Dense(self.features[-1])

    def __call__(self, x, return_features=False):
        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        if return_features:
            # Return last-layer features before final transformation
            features = x
            output = self.output_layer(features)
            return output, features
        else:
            return self.output_layer(x)


def _quad_form(phi, cov, zeta_squared):
    """Return f★ᵀ(cov + ζ²I)^{-1} f★ for each row f★ of ``phi``."""
    reg_cov = cov + zeta_squared * jnp.eye(cov.shape[0])
    sol = jnp.linalg.solve(reg_cov, phi.T)
    return np.asarray(jnp.sum(phi.T * sol, axis=0))


# Activations that can be exported to ONNX / OMLT, mapped to the ONNX op
_EXPORT_ACTIVATIONS = {"relu": "Relu", "tanh": "Tanh"}


def _per_output(value, j):
    """Pick output ``j`` from a scalar or per-output parameter."""
    return float(value) if np.isscalar(value) else float(value[j])


class LLPRRegressor(MIPExportMixin, BaseEstimator, RegressorMixin):
    """
    Last-Layer Prediction Rigidity Regressor with sklearn interface.

    The uncertainty is σ²★ = α² f★ᵀ(FᵀF + ζ²I)^{-1} f★, where f★ are the
    last-layer features of the network augmented with a constant 1 so that the
    output bias contributes to the uncertainty (otherwise a point where every
    last-layer ReLU is inactive would get σ = 0).

    Parameters
    ----------
    hidden_dims : tuple of int
        Dimensions of hidden layers (does not include output dim)
    activation : str, default='silu'
        Activation function ('silu', 'relu', 'tanh', 'gelu')
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer
    n_epochs : int, default=400
        Number of training epochs
    batch_size : int, default=32
        Batch size for training. The last partial batch is used, so datasets
        smaller than ``batch_size`` are trained in a single full batch.
    early_stopping_patience : int, default=100
        Epochs to wait for validation improvement before stopping
        (holdout calibration only)
    weight_decay : float, default=0.0
        L2 regularization strength
    alpha_squared : float or 'auto', default='auto'
        Calibration parameter for uncertainty scaling, in units of y².
        With 'auto' it is the NLL-optimal value in closed form.
    zeta_squared : float or 'auto', default='auto'
        Regularization parameter for covariance matrix. With 'auto' it is chosen
        by grid search on the NLL, relative to trace(FᵀF) / n_features.
    val_size : float, default=0.1
        Fraction of training data held out for early stopping and calibration
        (holdout calibration only)
    calibration : {'auto', 'holdout', 'cv'}, default='auto'
        How α² and ζ² are calibrated. 'holdout' uses a validation split of
        ``val_size``. 'cv' fits ``cv`` extra models on K folds, calibrates on
        the pooled out-of-fold residuals, and then trains the final model on all
        the data for ``n_epochs`` (no early stopping). 'auto' uses 'cv' when
        there are fewer than 100 samples, which is the usual case in active
        learning, and 'holdout' otherwise.
    cv : int, default=5
        Number of folds for 'cv' calibration (capped at the number of samples).
    standardize : bool, default=True
        Standardize X and y internally. Predictions and uncertainties are
        returned in the original units.
    warm_start : bool, default=False
        Initialize the network from the previous fit's parameters when the
        architecture is unchanged (useful when refitting in an active learning
        loop). Fold models in 'cv' calibration always start from scratch.
    verbose : bool, default=False
        Print early stopping and calibration information.
    random_state : int, default=42
        Random seed for reproducibility
    """

    def __init__(
        self,
        hidden_dims=(64, 64),
        activation="silu",
        learning_rate=1e-3,
        n_epochs=400,
        batch_size=32,
        early_stopping_patience=100,
        weight_decay=0.0,
        alpha_squared="auto",
        zeta_squared="auto",
        val_size=0.1,
        calibration="auto",
        cv=5,
        standardize=True,
        warm_start=False,
        verbose=False,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.alpha_squared = alpha_squared
        self.zeta_squared = zeta_squared
        self.val_size = val_size
        self.calibration = calibration
        self.cv = cv
        self.standardize = standardize
        self.warm_start = warm_start
        self.verbose = verbose
        self.random_state = random_state

    def _get_activation(self):
        """Get activation function from string."""
        activations = {"silu": nn.silu, "relu": nn.relu, "tanh": nn.tanh, "gelu": nn.gelu}
        return activations.get(self.activation, nn.silu)

    def _create_train_state(self, rng, input_dim, output_dim, init_params=None):
        """Create initial training state."""
        features = tuple(self.hidden_dims) + (output_dim,)
        model = MLP(features=features, activation=self._get_activation())

        # Initialize parameters
        if init_params is None:
            dummy_input = jnp.ones((1, input_dim))
            init_params = model.init(rng, dummy_input)

        # Create optimizer with optional weight decay
        if self.weight_decay > 0:
            optimizer = optax.adamw(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = optax.adam(learning_rate=self.learning_rate)

        return (
            train_state.TrainState.create(apply_fn=model.apply, params=init_params, tx=optimizer),
            model,
        )

    @staticmethod
    def _mse_loss(params, apply_fn, X, y):
        """Mean squared error loss."""
        predictions = apply_fn(params, X)
        return jnp.mean((predictions - y) ** 2)

    def _make_train_step(self):
        """Create a JIT-compiled training step function."""

        @jit
        def train_step(state, X_batch, y_batch):
            """Single training step."""

            def loss_fn(params):
                predictions = state.apply_fn(params, X_batch)
                return jnp.mean((predictions - y_batch) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        return train_step

    def _create_batches(self, X, y, rng):
        """Create shuffled batches, including the last partial batch."""
        n_samples = X.shape[0]
        perm = random.permutation(rng, n_samples)
        batch_size = max(1, min(self.batch_size, n_samples))

        for start_idx in range(0, n_samples, batch_size):
            idx = perm[start_idx : start_idx + batch_size]
            yield X[idx], y[idx]

    def _train(self, X_train, y_train, rng, X_val=None, y_val=None, init_params=None):
        """Train a network and return its parameters.

        If validation data is given, the parameters with the best validation
        loss are returned (with early stopping); otherwise the final ones.
        """
        rng, init_rng = random.split(rng)
        state, model = self._create_train_state(
            init_rng, X_train.shape[1], y_train.shape[1], init_params
        )
        self.model_ = model

        # Create JIT-compiled training step
        train_step_fn = self._make_train_step()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_params = state.params

        for epoch in range(self.n_epochs):
            # Training
            rng, batch_rng = random.split(rng)

            for X_batch, y_batch in self._create_batches(X_train, y_train, batch_rng):
                state, _ = train_step_fn(state, X_batch, y_batch)

            # Validation
            if X_val is not None:
                val_predictions = state.apply_fn(state.params, X_val)
                val_loss = jnp.mean((val_predictions - y_val) ** 2)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = state.params
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    if epoch > 50:  # Minimum training epochs
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            else:
                best_params = state.params

        return best_params

    def _features(self, params, X):
        """Return predictions and last-layer features augmented with a constant 1."""
        output, features = self.model_.apply(params, X, return_features=True)
        phi = jnp.concatenate([features, jnp.ones((features.shape[0], 1))], axis=1)
        return output, phi

    def _transform_X(self, X):
        return jnp.array((np.asarray(X, dtype=float) - self.X_mean_) / self.X_scale_)

    def fit(self, X, y):
        """
        Fit the neural network and compute last-layer covariance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values

        Returns
        -------
        self : object
            Fitted estimator
        """
        if not jax.config.read("jax_enable_x64"):
            warnings.warn(
                "JAX x64 mode is not enabled; LLPR will run in float32. "
                "The covariance inversion in LLPR is sensitive to ill-conditioning "
                "and benefits from float64. To enable, add this BEFORE importing "
                "pycse.sklearn.llpr_regressor (or any jax module):\n"
                "    import jax\n"
                "    jax.config.update('jax_enable_x64', True)",
                stacklevel=2,
            )

        if self.calibration not in ("auto", "holdout", "cv"):
            raise ValueError(
                f"Unknown calibration {self.calibration!r}. Choose from 'auto', 'holdout', 'cv'."
            )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Ensure y is 2D: (n_samples, n_outputs)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        self.n_outputs_ = y.shape[1]
        self.n_features_in_ = X.shape[1]
        # Training data range, the default input bounds for MIP/ONNX exports
        self.X_min_, self.X_max_ = X.min(axis=0), X.max(axis=0)

        # Internal standardization; everything below works in scaled units
        if self.standardize:
            self.X_mean_, self.X_scale_ = X.mean(axis=0), X.std(axis=0)
            self.y_mean_, self.y_scale_ = y.mean(axis=0), y.std(axis=0)
            self.X_scale_[self.X_scale_ == 0] = 1.0
            self.y_scale_[self.y_scale_ == 0] = 1.0
        else:
            self.X_mean_, self.X_scale_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
            self.y_mean_, self.y_scale_ = np.zeros(self.n_outputs_), np.ones(self.n_outputs_)

        Xs = (X - self.X_mean_) / self.X_scale_
        ys = (y - self.y_mean_) / self.y_scale_

        # Reuse the previous parameters if requested and the architecture matches
        signature = (self.n_features_in_, tuple(self.hidden_dims), self.n_outputs_)
        init_params = None
        if self.warm_start and getattr(self, "_fit_signature_", None) == signature:
            init_params = self.params_
        self._fit_signature_ = signature

        needs_calibration = self.alpha_squared == "auto" or self.zeta_squared == "auto"
        calibration = self.calibration
        if calibration == "auto":
            calibration = "cv" if n_samples < 100 else "holdout"
        self.calibration_ = calibration

        rng = random.PRNGKey(self.random_state)
        folds = []  # (residuals, features, covariance) used for calibration

        if calibration == "holdout":
            if self.val_size > 0 and n_samples >= 2:
                X_train, X_val, y_train, y_val = train_test_split(
                    Xs, ys, test_size=self.val_size, random_state=self.random_state
                )
                X_val, y_val = jnp.array(X_val), jnp.array(y_val)
            else:
                X_train, y_train = Xs, ys
                X_val, y_val = None, None

            X_train, y_train = jnp.array(X_train), jnp.array(y_train)
            self.params_ = self._train(X_train, y_train, rng, X_val, y_val, init_params)
            self._compute_covariance(X_train)

            if needs_calibration and X_val is not None:
                pred, phi = self._features(self.params_, X_val)
                folds.append((np.asarray(y_val - pred), phi, self.cov_matrix_))
        else:
            n_splits = min(self.cv, n_samples)
            if needs_calibration and n_splits >= 2:
                kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                for train_idx, test_idx in kfold.split(Xs):
                    rng, fold_rng = random.split(rng)
                    X_tr, y_tr = jnp.array(Xs[train_idx]), jnp.array(ys[train_idx])
                    params_k = self._train(X_tr, y_tr, fold_rng)
                    _, phi_tr = self._features(params_k, X_tr)
                    pred, phi = self._features(params_k, jnp.array(Xs[test_idx]))
                    folds.append((ys[test_idx] - np.asarray(pred), phi, phi_tr.T @ phi_tr))

            rng, final_rng = random.split(rng)
            X_all = jnp.array(Xs)
            self.params_ = self._train(X_all, jnp.array(ys), final_rng, init_params=init_params)
            self._compute_covariance(X_all)

        self._calibrate_uncertainty(folds)

        return self

    def _compute_covariance(self, X):
        """Compute the FᵀF covariance matrix from (scaled) training data."""
        _, phi = self._features(self.params_, X)
        self.n_features_ = phi.shape[1]
        self.cov_matrix_ = phi.T @ phi

    def _calibrate_uncertainty(self, folds):
        """
        Calibrate alpha_squared and zeta_squared on held-out residuals.

        ``folds`` is a list of ``(residuals, features, covariance)`` from a
        validation split or from cross-validation, all in standardized units.
        For each ζ² candidate the NLL-optimal α² = mean(r² / u) is used, where
        u = f★ᵀ(C + ζ²I)^{-1} f★, and the ζ² with the lowest NLL is kept. ζ² is
        searched relative to trace(C) / n_features so that it transfers between
        the fold models and the final model. With no folds, defaults are used.

        For multi-output models, each output is calibrated separately.
        """
        alpha_squared_list = []
        zeta_squared_list = []
        relative_zetas = np.logspace(-8, 2, 41)
        final_trace = float(jnp.trace(self.cov_matrix_)) / self.n_features_

        for j in range(self.n_outputs_):
            y_scale2 = self.y_scale_[j] ** 2
            best_nll, best_alpha, best_zeta = float("inf"), None, None

            if folds:
                r2 = np.concatenate([res[:, j] ** 2 for res, _, _ in folds])

                if self.zeta_squared == "auto":
                    candidates = relative_zetas
                else:
                    candidates = [_per_output(self.zeta_squared, j)]

                for zeta in candidates:
                    u = []
                    for _, phi, cov in folds:
                        if self.zeta_squared == "auto":
                            zeta_k = zeta * float(jnp.trace(cov)) / cov.shape[0]
                        else:
                            zeta_k = zeta
                        u.append(_quad_form(phi, cov, zeta_k))
                    u = np.concatenate(u)

                    # Skip if any variance is non-positive or NaN
                    if not np.all(np.isfinite(u)) or np.any(u <= 0):
                        continue

                    if self.alpha_squared == "auto":
                        alpha = max(float(np.mean(r2 / u)), 1e-12)
                    else:
                        alpha = _per_output(self.alpha_squared, j) / y_scale2

                    variances = alpha * u
                    nll = np.mean(0.5 * (r2 / variances + np.log(2 * np.pi * variances)))

                    if np.isfinite(nll) and nll < best_nll:
                        best_nll, best_alpha, best_zeta = nll, alpha, zeta

            # alpha in original units of y²; zeta in absolute units of FᵀF
            if self.alpha_squared != "auto":
                alpha_squared_list.append(_per_output(self.alpha_squared, j))
            elif best_alpha is not None:
                alpha_squared_list.append(best_alpha * y_scale2)
            else:
                alpha_squared_list.append(y_scale2)

            if self.zeta_squared != "auto":
                zeta_squared_list.append(_per_output(self.zeta_squared, j))
            elif best_zeta is not None:
                zeta_squared_list.append(best_zeta * final_trace)
            else:
                zeta_squared_list.append(1e-6)

            if self.verbose:
                print(
                    f"Output {j}: alpha²={alpha_squared_list[-1]:.2e}, "
                    f"zeta²={zeta_squared_list[-1]:.2e}, NLL={best_nll:.4f}"
                )

        # Store as arrays
        self.alpha_squared_ = np.array(alpha_squared_list)
        self.zeta_squared_ = np.array(zeta_squared_list)

    def _compute_uncertainties_batch(self, features, alpha_squared, zeta_squared):
        """
        Compute uncertainties for a batch of (augmented) features.
        σ²★ = α² f★ᵀ(FᵀF + ζ²I)^{-1} f★
        """
        return alpha_squared * _quad_form(features, self.cov_matrix_, zeta_squared)

    def predict(self, X, return_std=False):
        """
        Predict using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
        return_std : bool, default=False
            If True, return standard deviation alongside predictions.

        Returns
        -------
        y_pred : array of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values. 1D for single output, 2D for multi-output.
        y_std : array of shape (n_samples,) or (n_samples, n_outputs), optional
            Standard deviation (only if return_std=True).
        """
        if return_std:
            return self.predict_with_uncertainty(X, return_std=True)

        predictions = self.model_.apply(self.params_, self._transform_X(X))
        predictions = np.array(predictions) * self.y_scale_ + self.y_mean_

        # Squeeze single-output to 1D for backward compatibility
        if self.n_outputs_ == 1:
            predictions = predictions.squeeze(axis=1)

        return predictions

    def predict_with_uncertainty(self, X, return_std=True):
        """
        Predict with uncertainty estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
        return_std : bool, default=True
            If True, return standard deviation; if False, return variance

        Returns
        -------
        y_pred : array of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        uncertainty : array of shape (n_samples,) or (n_samples, n_outputs)
            Predicted uncertainties (std or variance)
        """
        # Get predictions and last-layer features
        predictions, features = self._features(self.params_, self._transform_X(X))
        predictions = np.array(predictions) * self.y_scale_ + self.y_mean_

        # Compute per-output uncertainties using each output's calibration params
        variances = np.zeros(predictions.shape)

        for j in range(self.n_outputs_):
            variances[:, j] = self._compute_uncertainties_batch(
                features, self.alpha_squared_[j], self.zeta_squared_[j]
            )

        # Squeeze single-output to 1D for backward compatibility
        if self.n_outputs_ == 1:
            predictions = predictions.squeeze(axis=1)
            variances = variances.squeeze(axis=1)

        if return_std:
            return predictions, np.sqrt(variances)
        else:
            return predictions, variances

    def get_dense_layers(self):
        """
        Return the network as dense layers in the original units of X and y.

        The internal standardization is folded into the first and last layers,
        so the layers map raw X to raw y. Only the mean prediction is exported,
        not the uncertainty.

        Returns
        -------
        layers : list of (W, b, activation)
            ``W`` has shape (n_in, n_out) so that a layer computes
            ``activation(x @ W + b)``. ``activation`` is the network activation
            for hidden layers and 'linear' for the output layer.
        """
        if self.activation not in _EXPORT_ACTIVATIONS:
            raise ValueError(
                f"Cannot export a network with activation={self.activation!r}. "
                f"Supported: {sorted(_EXPORT_ACTIVATIONS)} ('relu' gives an exact MILP)."
            )

        params = self.params_["params"]
        names = [f"hidden_layers_{i}" for i in range(len(self.hidden_dims))] + ["output_layer"]
        layers = []
        for i, name in enumerate(names):
            W = np.asarray(params[name]["kernel"], dtype=np.float64)
            b = np.asarray(params[name]["bias"], dtype=np.float64)
            if i == 0:
                # (x - mean) / scale @ W + b = x @ (W / scale) + (b - (mean / scale) @ W)
                b = b - (self.X_mean_ / self.X_scale_) @ W
                W = W / self.X_scale_[:, None]
            if name == "output_layer":
                W = W * self.y_scale_
                b = b * self.y_scale_ + self.y_mean_
            activation = "linear" if name == "output_layer" else self.activation
            layers.append((W, b, activation))
        return layers

    def mip_network(self):
        """Return the trained model as a ``MIPNetwork`` for MIP export (``to_pyomo()``).

        Requires ``activation='relu'``. Unlike ``to_omlt()``, the uncertainty is
        exported too: output j has the LLPR std
        ``sqrt(alpha_j^2 [f, 1]^T (F^T F + zeta_j^2 I)^{-1} [f, 1])`` on the
        last-layer features ``f`` (with the constant feature), a second-order cone.
        The internal standardization is folded in, so inputs and outputs are in
        original units. See ``pycse.sklearn.mip``.
        """
        if not hasattr(self, "params_"):
            raise ValueError("Fit the model before exporting it.")
        if self.activation != "relu":
            raise ValueError(
                "MIP export requires LLPRRegressor(activation='relu') (piecewise linear); "
                f"got activation={self.activation!r}."
            )

        params = self.params_["params"]
        layers = []
        for i in range(len(self.hidden_dims)):
            p = params[f"hidden_layers_{i}"]
            layers += [Affine(np.asarray(p["kernel"]), np.asarray(p["bias"])), ReLU()]
        p = params["output_layer"]
        layers.append(Affine(np.asarray(p["kernel"]), np.asarray(p["bias"])))

        std = None
        if hasattr(self, "cov_matrix_"):
            # The features are the last ReLU output (or the standardized inputs) with a
            # constant 1 appended: ||A [f, 1]||^2 = ||A[:, :-1] f + A[:, -1]||^2.
            tap = len(layers) - 2
            std = [
                StdSpec(tap=tap, A=spec.A[:, :-1], c=spec.A[:, -1], d=spec.d)
                for spec in llpr_std(self.cov_matrix_, self.alpha_squared_, self.zeta_squared_, tap)
            ]

        net = MIPNetwork(layers, std=std).scale_inputs(self.X_mean_, self.X_scale_)
        # alpha^2 is already in original units of y^2, so only the mean is rescaled.
        std, net.std = net.std, None
        net = net.scale_outputs(self.y_mean_, self.y_scale_)
        net.std = std
        return net

    def _default_input_bounds(self, input_bounds):
        if input_bounds is None:
            return np.asarray(self.X_min_, dtype=float), np.asarray(self.X_max_, dtype=float)
        lb, ub = input_bounds
        return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

    def to_onnx(self, path=None, opset=17):
        """
        Export the mean prediction network to ONNX.

        The graph is a sequential chain of Gemm and Relu (or Tanh) nodes in
        float64, in the original units of X and y. It can be loaded by
        ``discopt.ml.load_onnx``, OMLT's ONNX reader, or onnxruntime.

        Parameters
        ----------
        path : str or Path, optional
            If given, the model is also saved to this file.
        opset : int, default=17
            ONNX opset version.

        Returns
        -------
        model : onnx.ModelProto
        """
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except ImportError as e:
            raise ImportError("onnx is required for to_onnx. Install with: pip install onnx") from e

        nodes, initializers = [], []
        current = "X"
        for i, (W, b, activation) in enumerate(self.get_dense_layers()):
            initializers.append(numpy_helper.from_array(W, name=f"W{i}"))
            initializers.append(numpy_helper.from_array(b, name=f"b{i}"))
            out = f"z{i}"
            nodes.append(
                # alpha/beta/transB are written explicitly: OMLT's reader requires them
                helper.make_node(
                    "Gemm",
                    [current, f"W{i}", f"b{i}"],
                    [out],
                    name=f"dense{i}",
                    alpha=1.0,
                    beta=1.0,
                    transB=0,
                )
            )
            current = out
            if activation != "linear":
                out = f"a{i}"
                op = _EXPORT_ACTIVATIONS[activation]
                nodes.append(helper.make_node(op, [current], [out], name=f"{activation}{i}"))
                current = out
        # The output layer is linear, so its Gemm produces "y" directly (OMLT's
        # reader does not accept a trailing Identity node)
        nodes[-1].output[0] = "y"

        graph = helper.make_graph(
            nodes,
            "LLPRRegressor",
            [helper.make_tensor_value_info("X", TensorProto.DOUBLE, [None, self.n_features_in_])],
            [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [None, self.n_outputs_])],
            initializer=initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="pycse"
        )
        # Keep the IR version readable by older runtimes
        model.ir_version = min(model.ir_version, 8)
        onnx.checker.check_model(model)

        if path is not None:
            onnx.save(model, str(path))
        return model

    def to_omlt(self, input_bounds=None):
        """
        Export the mean prediction network as an OMLT ``NetworkDefinition``.

        Use it with a Pyomo ``OmltBlock`` and a formulation, e.g.
        ``ReluBigMFormulation`` for an exact MILP of a ReLU network::

            from omlt import OmltBlock
            from omlt.neuralnet import ReluBigMFormulation
            m.nn = OmltBlock()
            m.nn.build_formulation(ReluBigMFormulation(model.to_omlt()))
            # inputs: m.nn.inputs[i], outputs: m.nn.outputs[j]

        Parameters
        ----------
        input_bounds : tuple of (lb, ub) arrays, optional
            Bounds on each input in original units. Big-M formulations need
            them. Defaults to the range of the training data.

        Returns
        -------
        net : omlt.neuralnet.NetworkDefinition
        """
        try:
            from omlt.neuralnet import NetworkDefinition
            from omlt.neuralnet.layer import DenseLayer, InputLayer
        except ImportError as e:
            raise ImportError("omlt is required for to_omlt. Install with: pip install omlt") from e

        lb, ub = self._default_input_bounds(input_bounds)
        bounds = {i: (float(lb[i]), float(ub[i])) for i in range(self.n_features_in_)}
        net = NetworkDefinition(scaled_input_bounds=bounds, unscaled_input_bounds=bounds)

        previous = InputLayer([self.n_features_in_])
        net.add_layer(previous)
        for W, b, activation in self.get_dense_layers():
            layer = DenseLayer(
                previous.output_size, [W.shape[1]], weights=W, biases=b, activation=activation
            )
            net.add_layer(layer)
            net.add_edge(previous, layer)
            previous = layer
        return net

    def score(self, X, y):
        """
        Compute R² score.

        For multi-output, returns the average R² across all outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values

        Returns
        -------
        score : float
            R² score (averaged across outputs for multi-output)
        """
        y_pred = self.predict(X)
        y = np.array(y)

        if y.ndim == 1 and y_pred.ndim == 1:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)

        # Multi-output: average R² across outputs
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        r2_scores = []
        for j in range(y.shape[1]):
            ss_res = np.sum((y[:, j] - y_pred[:, j]) ** 2)
            ss_tot = np.sum((y[:, j] - np.mean(y[:, j])) ** 2)
            r2_scores.append(1 - (ss_res / ss_tot))

        return float(np.mean(r2_scores))


def compute_calibration_metrics(y_true, y_pred, y_std):
    """
    Compute calibration metrics for uncertainty estimates.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    y_std : array-like
        Predicted standard deviations

    Returns
    -------
    dict with keys:
        - rmse: Root mean squared error
        - nll: Negative log-likelihood
        - calibration_error: Average calibration error
        - fraction_within_1_sigma: Fraction of errors within 1σ
        - fraction_within_2_sigma: Fraction of errors within 2σ
        - fraction_within_3_sigma: Fraction of errors within 3σ
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # NLL (assuming Gaussian)
    nll = np.mean(0.5 * ((y_true - y_pred) ** 2 / y_std**2 + np.log(2 * np.pi * y_std**2)))

    # Calibration: check if errors are consistent with predicted uncertainties
    standardized_errors = (y_true - y_pred) / y_std

    # Expected fractions within 1, 2, 3 sigma for a Gaussian
    expected_fractions = [0.6827, 0.9545, 0.9973]
    actual_fractions = []

    for n_sigma in [1, 2, 3]:
        fraction = np.mean(np.abs(standardized_errors) <= n_sigma)
        actual_fractions.append(fraction)

    calibration_error = np.mean(np.abs(np.array(actual_fractions) - np.array(expected_fractions)))

    return {
        "rmse": rmse,
        "nll": nll,
        "calibration_error": calibration_error,
        "fraction_within_1_sigma": actual_fractions[0],
        "fraction_within_2_sigma": actual_fractions[1],
        "fraction_within_3_sigma": actual_fractions[2],
    }


if __name__ == "__main__":
    # Example: Simple 1D regression task
    from sklearn.datasets import make_regression
    import matplotlib.pyplot as plt

    # Generate synthetic data
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model
    model = LLPRRegressor(
        hidden_dims=(64, 64),
        activation="silu",
        learning_rate=1e-3,
        n_epochs=200,
        batch_size=32,
        early_stopping_patience=50,
        alpha_squared="auto",
        zeta_squared="auto",
        val_size=0.2,
        random_state=42,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Predict with uncertainties
    y_pred, y_std = model.predict_with_uncertainty(X_test)

    # Evaluate
    r2 = model.score(X_test, y_test)
    metrics = compute_calibration_metrics(y_test, y_pred, y_std)

    print(f"\nR² Score: {r2:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"NLL: {metrics['nll']:.4f}")
    print(f"Calibration Error: {metrics['calibration_error']:.4f}")
    print(f"Fraction within 1σ: {metrics['fraction_within_1_sigma']:.3f} (expected: 0.683)")
    print(f"Fraction within 2σ: {metrics['fraction_within_2_sigma']:.3f} (expected: 0.955)")

    # Plot predictions with uncertainty
    plt.figure(figsize=(10, 6))
    indices = np.argsort(y_test)
    plt.plot(y_test[indices], label="True", linewidth=2)
    plt.plot(y_pred[indices], label="Predicted", linewidth=2)
    plt.fill_between(
        range(len(y_test)),
        (y_pred - 2 * y_std)[indices],
        (y_pred + 2 * y_std)[indices],
        alpha=0.3,
        label="95% Confidence",
    )
    plt.xlabel("Sample (sorted by true value)")
    plt.ylabel("Target Value")
    plt.legend()
    plt.title("LLPR Predictions with Uncertainty")
    plt.tight_layout()
    plt.savefig("llpr_predictions.png", dpi=150)
    print("\nPlot saved as 'llpr_predictions.png'")
