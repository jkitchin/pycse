"""
Last-Layer Prediction Rigidity (LLPR) Regressor

Implementation of the prediction rigidity formalism from:
Bigi, F., Chong, S., Ceriotti, M., & Grasselli, F. (2024).
A prediction rigidity formalism for low-cost uncertainties in trained neural networks.
Machine Learning: Science and Technology, 5, 045018.

This module provides an sklearn-compatible implementation using Flax/JAX.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Callable
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import numpy as np


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


class LLPRRegressor(BaseEstimator, RegressorMixin):
    """
    Last-Layer Prediction Rigidity Regressor with sklearn interface.

    Parameters
    ----------
    hidden_dims : tuple of int
        Dimensions of hidden layers (does not include output dim)
    activation : str, default='silu'
        Activation function ('silu', 'relu', 'tanh')
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer
    n_epochs : int, default=400
        Number of training epochs
    batch_size : int, default=32
        Batch size for training
    early_stopping_patience : int, default=100
        Epochs to wait for validation improvement before stopping
    weight_decay : float, default=0.0
        L2 regularization strength
    alpha_squared : float or 'auto', default='auto'
        Calibration parameter for uncertainty scaling
    zeta_squared : float or 'auto', default='auto'
        Regularization parameter for covariance matrix
    val_size : float, default=0.1
        Fraction of training data to use for validation
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
        self.random_state = random_state

    def _get_activation(self):
        """Get activation function from string."""
        activations = {"silu": nn.silu, "relu": nn.relu, "tanh": nn.tanh, "gelu": nn.gelu}
        return activations.get(self.activation, nn.silu)

    def _create_train_state(self, rng, input_dim):
        """Create initial training state."""
        # Full architecture including output dimension (1 for regression)
        features = self.hidden_dims + (1,)
        model = MLP(features=features, activation=self._get_activation())

        # Initialize parameters
        dummy_input = jnp.ones((1, input_dim))
        params = model.init(rng, dummy_input)

        # Create optimizer with optional weight decay
        if self.weight_decay > 0:
            optimizer = optax.adamw(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = optax.adam(learning_rate=self.learning_rate)

        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        ), model

    @staticmethod
    def _mse_loss(params, apply_fn, X, y):
        """Mean squared error loss."""
        predictions = apply_fn(params, X).squeeze()
        return jnp.mean((predictions - y) ** 2)

    def _make_train_step(self):
        """Create a JIT-compiled training step function."""

        @jit
        def train_step(state, X_batch, y_batch):
            """Single training step."""

            def loss_fn(params):
                predictions = state.apply_fn(params, X_batch).squeeze()
                return jnp.mean((predictions - y_batch) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        return train_step

    def _create_batches(self, X, y, rng):
        """Create shuffled batches."""
        n_samples = X.shape[0]
        perm = random.permutation(rng, n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        n_batches = n_samples // self.batch_size
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

    def fit(self, X, y):
        """
        Fit the neural network and compute last-layer covariance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Convert to JAX arrays
        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32).squeeze()

        # Train/validation split
        if self.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                np.array(X), np.array(y), test_size=self.val_size, random_state=self.random_state
            )
            X_train = jnp.array(X_train)
            y_train = jnp.array(y_train)
            X_val = jnp.array(X_val)
            y_val = jnp.array(y_val)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Initialize model
        rng = random.PRNGKey(self.random_state)
        rng, init_rng = random.split(rng)

        state, model = self._create_train_state(init_rng, X.shape[1])
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
            epoch_losses = []

            for X_batch, y_batch in self._create_batches(X_train, y_train, batch_rng):
                state, loss = train_step_fn(state, X_batch, y_batch)
                epoch_losses.append(loss)

            # Validation
            if X_val is not None:
                val_predictions = state.apply_fn(state.params, X_val).squeeze()
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
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                best_params = state.params

        # Store best parameters
        self.params_ = best_params

        # Compute last-layer covariance matrix F^T F
        self._compute_covariance(X_train)

        # Calibrate uncertainty parameters if needed
        if X_val is not None and (self.alpha_squared == "auto" or self.zeta_squared == "auto"):
            self._calibrate_uncertainty(X_val, y_val)
        else:
            self.alpha_squared_ = 1.0 if self.alpha_squared == "auto" else self.alpha_squared
            self.zeta_squared_ = 1e-6 if self.zeta_squared == "auto" else self.zeta_squared

        return self

    def _compute_covariance(self, X):
        """Compute F^T F covariance matrix from training data."""
        # Extract last-layer features for all training samples
        _, features = self.model_.apply(self.params_, X, return_features=True)

        # Compute F^T F in batches to save memory
        self.n_features_ = features.shape[1]
        cov_matrix = jnp.zeros((self.n_features_, self.n_features_))

        batch_size = min(1000, X.shape[0])
        n_batches = (X.shape[0] + batch_size - 1) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, X.shape[0])
            features_batch = features[start_idx:end_idx]
            cov_matrix += features_batch.T @ features_batch

        self.cov_matrix_ = cov_matrix

    def _calibrate_uncertainty(self, X_val, y_val):
        """
        Calibrate alpha_squared and zeta_squared on validation set.
        Uses grid search to minimize validation NLL.
        """
        # Get predictions and features on validation set
        y_pred = self.predict(X_val)
        _, features = self.model_.apply(self.params_, X_val, return_features=True)

        # Grid search over hyperparameters
        alpha_candidates = jnp.logspace(-2, 2, 20)
        zeta_candidates = jnp.logspace(-8, 0, 20)

        best_nll = float("inf")
        best_alpha = 1.0
        best_zeta = 1e-6

        for alpha in alpha_candidates:
            for zeta in zeta_candidates:
                # Compute uncertainties
                variances = self._compute_uncertainties_batch(features, alpha, zeta)

                # Compute negative log-likelihood
                nll = jnp.mean(
                    0.5 * ((y_val - y_pred) ** 2 / variances + jnp.log(2 * jnp.pi * variances))
                )

                if nll < best_nll:
                    best_nll = nll
                    best_alpha = alpha
                    best_zeta = zeta

        self.alpha_squared_ = (
            float(best_alpha) if self.alpha_squared == "auto" else self.alpha_squared
        )
        self.zeta_squared_ = float(best_zeta) if self.zeta_squared == "auto" else self.zeta_squared

        print(
            f"Calibrated: alpha²={self.alpha_squared_:.2e}, zeta²={self.zeta_squared_:.2e}, NLL={best_nll:.4f}"
        )

    def _compute_uncertainties_batch(self, features, alpha_squared, zeta_squared):
        """
        Compute uncertainties for a batch of features.
        σ²★ = α² f★ᵀ(F^T F + ζ²I)^{-1} f★
        """
        # Regularized inverse covariance
        reg_cov = self.cov_matrix_ + zeta_squared * jnp.eye(self.n_features_)
        inv_cov = jnp.linalg.inv(reg_cov)

        # Vectorized computation: for each feature vector f★
        # uncertainty = α² * f★^T * inv_cov * f★
        @jit
        def compute_single_uncertainty(f):
            return alpha_squared * f.T @ inv_cov @ f

        uncertainties = vmap(compute_single_uncertainty)(features)

        return uncertainties

    def predict(self, X):
        """
        Predict using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted values
        """
        X = jnp.array(X, dtype=jnp.float32)
        predictions = self.model_.apply(self.params_, X)
        return np.array(predictions.squeeze())

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
        y_pred : array of shape (n_samples,)
            Predicted values
        uncertainty : array of shape (n_samples,)
            Predicted uncertainties (std or variance)
        """
        X = jnp.array(X, dtype=jnp.float32)

        # Get predictions and last-layer features
        predictions, features = self.model_.apply(self.params_, X, return_features=True)

        # Compute uncertainties
        variances = self._compute_uncertainties_batch(
            features, self.alpha_squared_, self.zeta_squared_
        )

        predictions = np.array(predictions.squeeze())

        if return_std:
            return predictions, np.array(jnp.sqrt(variances))
        else:
            return predictions, np.array(variances)

    def score(self, X, y):
        """
        Compute R² score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values

        Returns
        -------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        y = np.array(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


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
