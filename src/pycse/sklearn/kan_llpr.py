"""KAN with Last-Layer Prediction Rigidity (LLPR) uncertainty quantification.

Combines Kolmogorov-Arnold Networks with the prediction rigidity formalism:
- Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.
- Bigi, F., et al. (2024). A prediction rigidity formalism for low-cost
  uncertainties in trained neural networks. ML: Sci. Technol., 5, 045018.

LLPR provides principled uncertainty estimates by computing how "rigid" the
prediction is with respect to perturbations in the last-layer features.
This is more efficient than ensemble methods (single forward pass) and
provides well-calibrated uncertainties when properly tuned.

Example usage:

    import numpy as np
    from pycse.sklearn.kan_llpr import KANLLPR

    # Generate data
    X = np.linspace(0, 1, 100)[:, None]
    y = np.sin(2 * np.pi * X.ravel()) + 0.1 * np.random.randn(100)

    # Train KAN with LLPR
    model = KANLLPR(layers=(1, 8, 1), grid_size=5)
    model.fit(X, y)

    # Predict with uncertainty
    y_pred, y_std = model.predict_with_uncertainty(X)

Requires: flax, optax, jax, scikit-learn
"""

import os

import jax
from jax import jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from flax import linen as nn

from pycse.sklearn.optimizers import run_optimizer
from pycse.sklearn.kan import KANLayer

os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


class _KANNWithFeatures(nn.Module):
    """A KAN that can return last-layer features for LLPR.

    Attributes:
        layers: Tuple of layer sizes (input_dim, hidden1, ..., output_dim).
        grid_size: Number of grid intervals for B-splines in each layer.
        spline_order: Order of B-splines (default 3 for cubic).
        grid_range: Input range for spline normalization.
        base_activation: Activation for residual ('silu' or 'linear').
    """

    layers: tuple
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple = (-1.0, 1.0)
    base_activation: str = "silu"

    @nn.compact
    def __call__(self, x, return_features=False):
        """Forward pass through the KAN.

        Args:
            x: Input tensor, shape (batch_size, n_features).
            return_features: If True, return (output, last_layer_features).

        Returns:
            Output tensor, or (output, features) if return_features=True.
        """
        # Process through all but the last KAN layer
        for i in range(len(self.layers) - 2):
            in_dim = self.layers[i]
            out_dim = self.layers[i + 1]

            x = KANLayer(
                in_features=in_dim,
                out_features=out_dim,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                grid_range=self.grid_range,
                base_activation=self.base_activation,
            )(x)

        # Last layer (output layer)
        if len(self.layers) >= 2:
            in_dim = self.layers[-2]
            out_dim = self.layers[-1]

            # Store features before last layer
            features = x

            # Apply last layer
            output = KANLayer(
                in_features=in_dim,
                out_features=out_dim,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                grid_range=self.grid_range,
                base_activation=self.base_activation,
            )(x)

            if return_features:
                return output, features
            return output

        return x


class KANLLPR(BaseEstimator, RegressorMixin):
    """KAN with Last-Layer Prediction Rigidity for uncertainty quantification.

    This model combines Kolmogorov-Arnold Networks (learnable spline activations)
    with the LLPR formalism for principled uncertainty estimates. LLPR computes
    how "rigid" each prediction is based on the covariance structure of the
    last-layer features.

    Uncertainty formula:
        σ²(x★) = α² · f(x★)ᵀ · (FᵀF + ζ²I)⁻¹ · f(x★)

    where:
        - f(x★) are the last-layer features for input x★
        - F is the matrix of last-layer features from training data
        - α² scales the uncertainty magnitude
        - ζ² regularizes the covariance inversion

    Parameters
    ----------
    layers : tuple of int
        Network architecture. layers[0] is input dim, layers[-1] is output dim.
        Example: (5, 16, 8, 1) for 5 inputs, two hidden layers, 1 output.
    grid_size : int, default=5
        Number of grid intervals for B-splines. More = more expressive.
    spline_order : int, default=3
        Order of B-spline (3 = cubic). Higher = smoother.
    grid_range : tuple, default=(-2.0, 2.0)
        Range for input normalization to splines.
    seed : int, default=42
        Random seed for reproducibility.
    optimizer : str, default='bfgs'
        Optimization algorithm ('bfgs', 'lbfgs', 'adam', 'sgd').
    l1_spline : float, default=0.0
        L1 regularization on spline coefficients.
    l1_activation : float, default=0.0
        L1 regularization on activation outputs.
    entropy_reg : float, default=0.0
        Entropy regularization for sharper activations.
    base_activation : str, default='silu'
        Base activation for residual connection ('silu' or 'linear').
    alpha_squared : float or 'auto', default='auto'
        Uncertainty scaling parameter. If 'auto', calibrated on validation set.
    zeta_squared : float or 'auto', default='auto'
        Covariance regularization. If 'auto', calibrated on validation set.
    val_size : float, default=0.1
        Fraction of training data for validation/calibration.

    Attributes
    ----------
    params_ : dict
        Trained model parameters.
    cov_matrix_ : ndarray
        FᵀF covariance matrix from training features.
    alpha_squared_ : float
        Calibrated uncertainty scale.
    zeta_squared_ : float
        Calibrated regularization.
    """

    def __init__(
        self,
        layers,
        grid_size=5,
        spline_order=3,
        grid_range=(-2.0, 2.0),
        seed=42,
        optimizer="bfgs",
        l1_spline=0.0,
        l1_activation=0.0,
        entropy_reg=0.0,
        base_activation="silu",
        alpha_squared="auto",
        zeta_squared="auto",
        val_size=0.1,
    ):
        self.layers = layers
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        self.optimizer = optimizer.lower()
        self.l1_spline = l1_spline
        self.l1_activation = l1_activation
        self.entropy_reg = entropy_reg
        self.base_activation = base_activation
        self.alpha_squared = alpha_squared
        self.zeta_squared = zeta_squared
        self.val_size = val_size
        self.n_outputs = layers[-1]

        # Create the network
        self.nn = _KANNWithFeatures(
            layers=layers,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
            base_activation=base_activation,
        )

        # Normalization parameters (set during fit)
        self.X_mean_ = None
        self.X_std_ = None
        self.y_mean_ = None
        self.y_std_ = None

    def _normalize_X(self, X):
        """Normalize input features."""
        if self.X_mean_ is not None and self.X_std_ is not None:
            return (X - self.X_mean_) / (self.X_std_ + 1e-8)
        return X

    def _normalize_y(self, y):
        """Normalize target values."""
        if self.y_mean_ is not None and self.y_std_ is not None:
            return (y - self.y_mean_) / (self.y_std_ + 1e-8)
        return y

    def _denormalize_y(self, y):
        """Denormalize predictions."""
        if self.y_mean_ is not None and self.y_std_ is not None:
            return y * self.y_std_ + self.y_mean_
        return y

    def _denormalize_std(self, std):
        """Denormalize standard deviations."""
        if self.y_std_ is not None:
            return std * self.y_std_
        return std

    def fit(self, X, y, **kwargs):
        """Fit the KAN model and compute LLPR covariance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        **kwargs : dict
            Additional arguments for optimizer:
            - maxiter: Maximum iterations (default: 1500)
            - tol: Convergence tolerance (default: 1e-3)

        Returns
        -------
        self : KANLLPR
            Fitted model.
        """
        X = jnp.atleast_2d(X)
        y = jnp.asarray(y)

        # Handle multi-output targets
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.shape[1] != self.n_outputs:
            raise ValueError(
                f"Target has {y.shape[1]} outputs but model expects {self.n_outputs}. "
                f"Set layers[-1]={y.shape[1]} to match."
            )

        # Train/validation split for calibration
        if self.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                np.array(X), np.array(y), test_size=self.val_size, random_state=self.seed
            )
            X_train = jnp.array(X_train)
            y_train = jnp.array(y_train)
            X_val = jnp.array(X_val)
            y_val = jnp.array(y_val)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Store normalization parameters
        self.X_mean_ = jnp.mean(X_train, axis=0)
        self.X_std_ = jnp.std(X_train, axis=0)
        self.y_mean_ = jnp.mean(y_train, axis=0)
        self.y_std_ = jnp.std(y_train, axis=0)

        # Normalize data
        X_norm = self._normalize_X(X_train)
        y_norm = self._normalize_y(y_train)

        # Initialize parameters
        params = self.nn.init(self.key, X_norm)

        # Store regularization params for closure
        l1_spline = self.l1_spline
        l1_activation = self.l1_activation
        entropy_reg = self.entropy_reg

        @jit
        def objective(pars):
            """Loss function with optional regularization."""
            pY = self.nn.apply(pars, X_norm)

            # MSE loss
            errs = y_norm - pY
            loss = jnp.mean(errs**2)

            # Add regularization terms
            reg_loss = 0.0

            if l1_spline > 0 or entropy_reg > 0:
                for key in pars["params"]:
                    if key.startswith("KANLayer_"):
                        layer_params = pars["params"][key]

                        if l1_spline > 0 and "spline_weight" in layer_params:
                            spline_w = layer_params["spline_weight"]
                            reg_loss = reg_loss + l1_spline * jnp.mean(jnp.abs(spline_w))

                        if entropy_reg > 0 and "spline_weight" in layer_params:
                            spline_w = layer_params["spline_weight"]
                            w_abs = jnp.abs(spline_w) + 1e-8
                            w_norm = w_abs / jnp.sum(w_abs, axis=-1, keepdims=True)
                            entropy = -jnp.sum(w_norm * jnp.log(w_norm), axis=-1)
                            reg_loss = reg_loss + entropy_reg * jnp.mean(entropy)

            if l1_activation > 0:
                reg_loss = reg_loss + l1_activation * jnp.mean(jnp.abs(pY))

            return loss + reg_loss

        # Run optimization
        maxiter = kwargs.pop("maxiter", 300)
        tol = kwargs.pop("tol", 1e-3)

        self.optpars, self.state = run_optimizer(
            self.optimizer, objective, params, maxiter=maxiter, tol=tol, **kwargs
        )

        # Compute LLPR covariance matrix from training features
        self._compute_covariance(X_train)

        # Calibrate uncertainty parameters
        if X_val is not None and (self.alpha_squared == "auto" or self.zeta_squared == "auto"):
            self._calibrate_uncertainty(X_val, y_val)
        else:
            # Set default values as arrays (one per output)
            if self.alpha_squared == "auto":
                self.alpha_squared_ = jnp.ones(self.n_outputs)
            elif np.isscalar(self.alpha_squared):
                self.alpha_squared_ = jnp.full(self.n_outputs, self.alpha_squared)
            else:
                self.alpha_squared_ = jnp.array(self.alpha_squared)

            if self.zeta_squared == "auto":
                self.zeta_squared_ = jnp.full(self.n_outputs, 1e-6)
            elif np.isscalar(self.zeta_squared):
                self.zeta_squared_ = jnp.full(self.n_outputs, self.zeta_squared)
            else:
                self.zeta_squared_ = jnp.array(self.zeta_squared)

        return self

    def _compute_covariance(self, X):
        """Compute FᵀF covariance matrix from training data.

        Parameters
        ----------
        X : array-like
            Training features.
        """
        X_norm = self._normalize_X(X)

        # Extract last-layer features
        _, features = self.nn.apply(self.optpars, X_norm, return_features=True)

        # Store feature dimension
        self.n_features_ = features.shape[1]

        # Compute FᵀF in batches to save memory
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
        """Calibrate α² and ζ² on validation set using grid search.

        For multi-output models, each output is calibrated separately.

        Parameters
        ----------
        X_val : array-like
            Validation features.
        y_val : array-like
            Validation targets.
        """
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val)

        # Ensure y_val is 2D
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)

        # Get predictions and features
        X_norm = self._normalize_X(X_val)
        predictions_norm, features = self.nn.apply(self.optpars, X_norm, return_features=True)

        # Normalize y_val for comparison
        y_val_norm = self._normalize_y(y_val)

        # Grid search over hyperparameters
        alpha_candidates = jnp.logspace(-2, 2, 20)
        zeta_candidates = jnp.logspace(-8, 0, 20)

        # Calibrate each output separately
        alpha_squared_list = []
        zeta_squared_list = []

        for j in range(self.n_outputs):
            best_nll = float("inf")
            best_alpha = 1.0
            best_zeta = 1e-6

            y_j = y_val_norm[:, j]
            pred_j = predictions_norm[:, j]

            for alpha in alpha_candidates:
                for zeta in zeta_candidates:
                    # Compute uncertainties (same features, but calibration differs per output)
                    variances = self._compute_uncertainties_batch(features, alpha, zeta)

                    # Compute negative log-likelihood for this output
                    nll = jnp.mean(
                        0.5 * ((y_j - pred_j) ** 2 / variances + jnp.log(2 * jnp.pi * variances))
                    )

                    if jnp.isfinite(nll) and nll < best_nll:
                        best_nll = nll
                        best_alpha = alpha
                        best_zeta = zeta

            if self.alpha_squared == "auto":
                alpha_squared_list.append(float(best_alpha))
            else:
                alpha_squared_list.append(
                    self.alpha_squared if np.isscalar(self.alpha_squared) else self.alpha_squared[j]
                )

            if self.zeta_squared == "auto":
                zeta_squared_list.append(float(best_zeta))
            else:
                zeta_squared_list.append(
                    self.zeta_squared if np.isscalar(self.zeta_squared) else self.zeta_squared[j]
                )

            if self.n_outputs > 1:
                print(
                    f"Output {j}: α²={alpha_squared_list[-1]:.2e}, "
                    f"ζ²={zeta_squared_list[-1]:.2e}, NLL={best_nll:.4f}"
                )

        # Store as arrays
        self.alpha_squared_ = jnp.array(alpha_squared_list)
        self.zeta_squared_ = jnp.array(zeta_squared_list)

        if self.n_outputs == 1:
            print(
                f"Calibrated: α²={self.alpha_squared_[0]:.2e}, "
                f"ζ²={self.zeta_squared_[0]:.2e}, NLL={best_nll:.4f}"
            )

    def _compute_uncertainties_batch(self, features, alpha_squared, zeta_squared):
        """Compute LLPR uncertainties for a batch of features.

        σ²(x★) = α² · f★ᵀ · (FᵀF + ζ²I)⁻¹ · f★

        Parameters
        ----------
        features : array-like
            Last-layer features, shape (n_samples, n_features).
        alpha_squared : float
            Uncertainty scaling parameter.
        zeta_squared : float
            Regularization parameter.

        Returns
        -------
        uncertainties : array
            Variance estimates, shape (n_samples,).
        """
        # Regularized inverse covariance
        reg_cov = self.cov_matrix_ + zeta_squared * jnp.eye(self.n_features_)
        inv_cov = jnp.linalg.inv(reg_cov)

        # Vectorized computation
        @jit
        def compute_single_uncertainty(f):
            return alpha_squared * f.T @ inv_cov @ f

        uncertainties = vmap(compute_single_uncertainty)(features)

        return uncertainties

    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : array of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
        X = jnp.atleast_2d(X)
        X_norm = self._normalize_X(X)
        predictions_norm = self.nn.apply(self.optpars, X_norm)
        predictions = self._denormalize_y(predictions_norm)

        # Squeeze if single output
        if self.n_outputs == 1:
            predictions = predictions.squeeze(axis=1)

        return np.array(predictions)

    def predict_with_uncertainty(self, X, return_std=True):
        """Predict with LLPR uncertainty estimates.

        For multi-output models, each output has its own uncertainty estimate
        based on its individually calibrated α² and ζ² parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        return_std : bool, default=True
            If True, return standard deviation; if False, return variance.

        Returns
        -------
        y_pred : array of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        uncertainty : array of shape (n_samples,) or (n_samples, n_outputs)
            Uncertainty estimates (std or variance) for each output.
        """
        X = jnp.atleast_2d(X)
        X_norm = self._normalize_X(X)

        # Get predictions and features
        predictions_norm, features = self.nn.apply(self.optpars, X_norm, return_features=True)

        # Compute uncertainties for each output with its own calibration
        n_samples = X.shape[0]
        variances_norm = jnp.zeros((n_samples, self.n_outputs))

        for j in range(self.n_outputs):
            alpha_j = self.alpha_squared_[j]
            zeta_j = self.zeta_squared_[j]
            var_j = self._compute_uncertainties_batch(features, alpha_j, zeta_j)
            variances_norm = variances_norm.at[:, j].set(var_j)

        # Denormalize predictions
        predictions = self._denormalize_y(predictions_norm)

        # Denormalize uncertainties (variance scales with y_std² per output)
        if self.y_std_ is not None:
            # Scale each output's variance by its own y_std²
            variances = variances_norm * (self.y_std_**2)
        else:
            variances = variances_norm

        # Squeeze if single output
        if self.n_outputs == 1:
            predictions = predictions.squeeze(axis=1)
            variances = variances.squeeze(axis=1)

        predictions = np.array(predictions)
        variances = np.array(variances)

        if return_std:
            return predictions, np.sqrt(variances)
        else:
            return predictions, variances

    def score(self, X, y):
        """Compute R² score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        score : float
            R² score.
        """
        y_pred = self.predict(X)
        y = np.array(y).ravel()
        y_pred = np.array(y_pred).ravel()

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def report(self):
        """Print model diagnostics."""
        print("KANLLPR Model Report:")
        print(f"  Architecture: {self.layers}")
        print(f"  Grid size: {self.grid_size}")
        print(f"  Spline order: {self.spline_order}")
        print(f"  Optimizer: {self.optimizer}")

        if self.l1_spline > 0 or self.l1_activation > 0 or self.entropy_reg > 0:
            print("  Regularization:")
            if self.l1_spline > 0:
                print(f"    L1 spline: {self.l1_spline}")
            if self.l1_activation > 0:
                print(f"    L1 activation: {self.l1_activation}")
            if self.entropy_reg > 0:
                print(f"    Entropy: {self.entropy_reg}")

        if hasattr(self, "state"):
            if hasattr(self.state, "iter_num"):
                print(f"  Iterations: {self.state.iter_num}")
            if hasattr(self.state, "value"):
                print(f"  Final loss: {self.state.value:.6f}")
            if hasattr(self.state, "converged"):
                print(f"  Converged: {self.state.converged}")

        if hasattr(self, "alpha_squared_"):
            if self.n_outputs == 1:
                print(f"  LLPR α²: {self.alpha_squared_[0]:.2e}")
                print(f"  LLPR ζ²: {self.zeta_squared_[0]:.2e}")
            else:
                print("  LLPR calibration (per output):")
                for j in range(self.n_outputs):
                    print(
                        f"    Output {j}: α²={self.alpha_squared_[j]:.2e}, ζ²={self.zeta_squared_[j]:.2e}"
                    )
            print(f"  Feature dimension: {self.n_features_}")

    def plot(self, X, y, ax=None):
        """Visualize predictions with LLPR uncertainty bands.

        Parameters
        ----------
        X : array-like
            Input features (works best with 1D input).
        y : array-like
            True target values.
        ax : matplotlib axis, optional
            Axis to plot on.

        Returns
        -------
        fig : matplotlib figure
        """
        if ax is None:
            ax = plt.gca()

        X = np.asarray(X)
        y = np.asarray(y)

        # Get predictions with uncertainty
        mp, se = self.predict_with_uncertainty(X, return_std=True)
        mp = np.asarray(mp)
        se = np.asarray(se)

        # For 1D input, sort for line plot
        X_plot = X.ravel()
        sort_idx = np.argsort(X_plot)
        X_sorted = X_plot[sort_idx]
        mp_sorted = mp[sort_idx]
        se_sorted = se[sort_idx]

        # Plot uncertainty band
        ax.fill_between(
            X_sorted,
            mp_sorted - 2 * se_sorted,
            mp_sorted + 2 * se_sorted,
            alpha=0.3,
            color="red",
            label="±2σ (95% CI)",
            zorder=1,
        )

        ax.fill_between(
            X_sorted,
            mp_sorted - se_sorted,
            mp_sorted + se_sorted,
            alpha=0.5,
            color="red",
            label="±1σ (68% CI)",
            zorder=2,
        )

        # Plot mean prediction
        ax.plot(X_sorted, mp_sorted, "r-", label="KANLLPR prediction", linewidth=2.5, zorder=3)

        # Plot data
        ax.plot(X_plot, y, "b.", label="data", alpha=0.7, markersize=8, zorder=4)

        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title(f"KANLLPR Predictions (grid={self.grid_size}, layers={self.layers})")
        ax.grid(True, alpha=0.3)

        return plt.gcf()

    def uncertainty_metrics(self, X, y):
        """Compute uncertainty quantification metrics.

        For multi-output models, metrics are computed for each output
        and also aggregated across all outputs.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            True target values.

        Returns
        -------
        dict with keys:
            - rmse: Root mean squared error (averaged across outputs)
            - mae: Mean absolute error (averaged across outputs)
            - nll: Negative log-likelihood (averaged across outputs)
            - miscalibration_area: Average calibration error
            - z_score_mean: Mean of standardized residuals (ideal: 0)
            - z_score_std: Std of standardized residuals (ideal: 1)
            - fraction_within_1_sigma: Fraction of errors within 1σ
            - fraction_within_2_sigma: Fraction of errors within 2σ
            - fraction_within_3_sigma: Fraction of errors within 3σ
            - per_output: dict with metrics for each output (if n_outputs > 1)
        """
        mp, se = self.predict_with_uncertainty(X, return_std=True)
        y = np.asarray(y)
        mp = np.asarray(mp)
        se = np.asarray(se)

        # Ensure 2D for consistent handling
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if mp.ndim == 1:
            mp = mp.reshape(-1, 1)
        if se.ndim == 1:
            se = se.reshape(-1, 1)

        # Guard against zero uncertainties
        se = np.maximum(se, 1e-10)

        # Compute per-output metrics
        per_output_metrics = []
        all_z_scores = []

        for j in range(self.n_outputs):
            errs_j = y[:, j] - mp[:, j]
            se_j = se[:, j]

            rmse_j = float(np.sqrt(np.mean(errs_j**2)))
            mae_j = float(np.mean(np.abs(errs_j)))
            nll_j = 0.5 * np.mean(errs_j**2 / se_j**2 + np.log(2 * np.pi * se_j**2))
            z_scores_j = errs_j / se_j
            all_z_scores.extend(z_scores_j)

            per_output_metrics.append(
                {
                    "rmse": rmse_j,
                    "mae": mae_j,
                    "nll": float(nll_j),
                    "z_score_mean": float(np.mean(z_scores_j)),
                    "z_score_std": float(np.std(z_scores_j)),
                }
            )

        # Aggregate metrics
        all_z_scores = np.array(all_z_scores)
        rmse = np.mean([m["rmse"] for m in per_output_metrics])
        mae = np.mean([m["mae"] for m in per_output_metrics])
        nll = np.mean([m["nll"] for m in per_output_metrics])
        z_mean = np.mean(all_z_scores)
        z_std = np.std(all_z_scores)

        # Miscalibration area (using all z-scores)
        sorted_z = np.sort(all_z_scores)
        empirical_cdf = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
        from scipy import stats

        theoretical_cdf = stats.norm.cdf(sorted_z)
        miscalibration_area = np.mean(np.abs(empirical_cdf - theoretical_cdf))

        # Calibration fractions
        actual_fractions = []
        for n_sigma in [1, 2, 3]:
            fraction = np.mean(np.abs(all_z_scores) <= n_sigma)
            actual_fractions.append(fraction)

        result = {
            "rmse": float(rmse),
            "mae": float(mae),
            "nll": float(nll),
            "miscalibration_area": float(miscalibration_area),
            "z_score_mean": float(z_mean),
            "z_score_std": float(z_std),
            "fraction_within_1_sigma": actual_fractions[0],
            "fraction_within_2_sigma": actual_fractions[1],
            "fraction_within_3_sigma": actual_fractions[2],
        }

        if self.n_outputs > 1:
            result["per_output"] = per_output_metrics

        return result

    def print_metrics(self, X, y):
        """Print uncertainty metrics in human-readable format.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            True target values.
        """
        metrics = self.uncertainty_metrics(X, y)

        print("\n" + "=" * 50)
        print("KANLLPR UNCERTAINTY QUANTIFICATION METRICS")
        print("=" * 50)

        if self.n_outputs > 1:
            print(f"(Aggregated over {self.n_outputs} outputs)")

        print("\nPrediction Accuracy:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")

        print("\nUncertainty Quality:")
        print(f"  NLL: {metrics['nll']:.6f}")
        print(f"  Miscalibration Area: {metrics['miscalibration_area']:.6f}")

        print("\nCalibration Diagnostics:")
        print(f"  Z-score mean: {metrics['z_score_mean']:.4f} (ideal: 0)")
        print(f"  Z-score std:  {metrics['z_score_std']:.4f} (ideal: 1)")

        print("\nCoverage:")
        print(f"  Within 1σ: {metrics['fraction_within_1_sigma']:.3f} (expected: 0.683)")
        print(f"  Within 2σ: {metrics['fraction_within_2_sigma']:.3f} (expected: 0.955)")
        print(f"  Within 3σ: {metrics['fraction_within_3_sigma']:.3f} (expected: 0.997)")

        # Per-output metrics for multi-output models
        if self.n_outputs > 1 and "per_output" in metrics:
            print("\nPer-Output Metrics:")
            for j, m in enumerate(metrics["per_output"]):
                print(
                    f"  Output {j}: RMSE={m['rmse']:.4f}, NLL={m['nll']:.4f}, "
                    f"z_mean={m['z_score_mean']:.3f}, z_std={m['z_score_std']:.3f}"
                )

        print("=" * 50 + "\n")

    def __call__(self, X, return_std=False):
        """Execute the model (alternative interface to predict).

        Parameters
        ----------
        X : array-like
            Input features.
        return_std : bool, default=False
            If True, return uncertainties.

        Returns
        -------
        Predictions (and uncertainties if requested).
        """
        if not hasattr(self, "optpars"):
            raise Exception("You need to fit the model first.")

        if return_std:
            return self.predict_with_uncertainty(X, return_std=True)

        return self.predict(X)


def compute_calibration_metrics(y_true, y_pred, y_std):
    """Compute calibration metrics for uncertainty estimates.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    y_std : array-like
        Predicted standard deviations.

    Returns
    -------
    dict with calibration metrics.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    y_std = np.array(y_std).ravel()

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # NLL (assuming Gaussian)
    y_std = np.maximum(y_std, 1e-10)
    nll = np.mean(0.5 * ((y_true - y_pred) ** 2 / y_std**2 + np.log(2 * np.pi * y_std**2)))

    # Calibration
    standardized_errors = (y_true - y_pred) / y_std

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
    # Example: 1D regression with heteroscedastic noise
    import matplotlib.pyplot as plt

    np.random.seed(42)

    # Generate data with varying noise
    X = np.linspace(0, 2 * np.pi, 100)[:, None]
    noise = 0.1 + 0.2 * np.abs(np.sin(X.ravel()))  # Heteroscedastic noise
    y = np.sin(X.ravel()) + noise * np.random.randn(100)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model
    print("Training KANLLPR model...")
    model = KANLLPR(
        layers=(1, 8, 1),
        grid_size=5,
        spline_order=3,
        optimizer="bfgs",
        alpha_squared="auto",
        zeta_squared="auto",
        val_size=0.2,
        seed=42,
    )

    model.fit(X_train, y_train, maxiter=5000)

    # Report
    model.report()

    # Evaluate
    print(f"\nR² Score: {model.score(X_test, y_test):.4f}")
    model.print_metrics(X_test, y_test)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    model.plot(X_test, y_test, ax=ax)
    plt.tight_layout()
    plt.savefig("kanllpr_predictions.png", dpi=150)
    print("Plot saved as 'kanllpr_predictions.png'")
