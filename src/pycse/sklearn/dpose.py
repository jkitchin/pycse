"""A DPOSE (Direct Propagation of Shallow Ensembles) Neural Network model in JAX.

Implementation based on:
Kellner, M., & Ceriotti, M. (2024). Uncertainty quantification by direct
propagation of shallow ensembles. Machine Learning: Science and Technology, 5(3), 035006.

Key features:
- Shallow ensemble architecture (only last layer differs across members)
- CRPS loss for robust, calibrated uncertainty estimates (default)
- Alternative NLL or MSE losses available
- Post-hoc calibration on validation set
- Ensemble propagation for derived quantities

Example usage:

    import jax
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate heteroscedastic data
    key = jax.random.PRNGKey(19)
    x = np.linspace(0, 1, 100)[:, None]
    noise_level = 0.01 + 0.1 * x.ravel()  # Increasing noise
    y = x.ravel()**(1/3) + noise_level * jax.random.normal(key, (100,))

    # Split into train/validation
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train with DPOSE (uses CRPS loss and BFGS optimizer by default)
    from pycse.sklearn.dpose import DPOSE
    model = DPOSE(layers=(1, 15, 32))
    model.fit(x_train, y_train, val_X=x_val, val_y=y_val)

    # Or use a different optimizer (e.g., Adam)
    model_adam = DPOSE(layers=(1, 15, 32), optimizer='adam')
    model_adam.fit(x_train, y_train, val_X=x_val, val_y=y_val, learning_rate=1e-3)

    # Or use Muon optimizer (state-of-the-art 2024)
    model_muon = DPOSE(layers=(1, 15, 32), optimizer='muon')
    model_muon.fit(x_train, y_train, val_X=x_val, val_y=y_val, learning_rate=0.02)

    # Get predictions with uncertainty
    y_pred, y_std = model.predict(x, return_std=True)

    # Visualize
    model.plot(x, y, distribution=True)

    # For uncertainty propagation on derived quantities
    ensemble_preds = model.predict_ensemble(x)  # (n_samples, n_ensemble)
    # Apply any function f to ensemble members
    z_ensemble = f(ensemble_preds)
    z_mean = z_ensemble.mean(axis=1)
    z_std = z_ensemble.std(axis=1)

Requires: flax, jaxopt, jax, scikit-learn
"""

import os
import jax


from jax import jit
import jax.numpy as np
from jax import value_and_grad
import jaxopt
import optax
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from flax import linen as nn
from flax.linen.initializers import xavier_uniform

os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


class _NN(nn.Module):
    """A flax neural network.

    layers: a Tuple of integers specifying the network architecture.
    - layers[0]: Input dimension (number of features)
    - layers[1:-1]: Hidden layer sizes
    - layers[-1]: Output dimension (ensemble size)

    Example: layers=(5, 20, 32) creates:
    - Input: 5 features
    - Hidden: 20 neurons with activation
    - Output: 32 ensemble members (no activation)
    """

    layers: tuple
    activation: callable

    @nn.compact
    def __call__(self, x):

        # Hidden layers (skip first element which is input dimension)
        for i in self.layers[1:-1]:
            x = nn.Dense(i, kernel_init=xavier_uniform())(x)
            x = self.activation(x)

        # Linear last layer where each row is a set of predictions
        # The mean on axis=1 is the prediction
        x = nn.Dense(self.layers[-1])(x)
        return x


class DPOSE(BaseEstimator, RegressorMixin):
    """DPOSE: Direct Propagation of Shallow Ensembles.

    A shallow ensemble neural network where only the last layer differs across
    ensemble members. Provides calibrated uncertainty estimates through CRPS or
    NLL training.

    The last element of `layers` determines the ensemble size (n_ensemble).
    For example, layers=(5, 10, 32) creates:
    - Input layer: 5 features
    - Hidden layer: 10 neurons
    - Output layer: 32 ensemble members

    Key Features:
    - CRPS loss (default): Robust, works out-of-the-box
    - NLL loss: Automatically pre-trains with MSE for robustness
    - Post-hoc calibration on validation set
    - Uncertainty propagation through ensemble members
    """

    def __init__(
        self,
        layers,
        activation=nn.relu,
        seed=19,
        loss_type="crps",
        min_sigma=1e-3,
        optimizer="bfgs",
    ):
        """Initialize a DPOSE model.

        Args:
            layers: Tuple of integers for neurons in each layer.
                   The last value is the ensemble size (recommended: 16-64).
            activation: Activation function for hidden layers (default: ReLU).
            seed: Random seed for weight initialization.
            loss_type: Loss function - 'crps', 'nll', or 'mse' (default: 'crps').
                      - 'crps': Continuous ranked probability score (Kellner Eq. 18) - RECOMMENDED
                                More robust, prevents uncertainty inflation, works out-of-the-box.
                      - 'nll': Negative log-likelihood (Kellner Eq. 6)
                               Can fail without pre-training or normalization (see WHY_NLL_FAILS.md).
                      - 'mse': Mean squared error (no uncertainty training)
            min_sigma: Minimum standard deviation for numerical stability (default: 1e-3).
                      Prevents division by zero when ensemble members are nearly identical.
            optimizer: Optimization algorithm (default: 'bfgs'). Options:
                      - 'bfgs': BFGS (quasi-Newton, recommended for smooth objectives)
                      - 'lbfgs': Limited-memory BFGS (for larger problems)
                      - 'adam': Adam (adaptive learning rate)
                      - 'sgd': Stochastic gradient descent
                      - 'muon': Muon (orthogonalized momentum, state-of-the-art 2024)
                      - 'lbfgsb': L-BFGS-B (with box constraints)
                      - 'nonlinear_cg': Nonlinear conjugate gradient
                      - 'gradient_descent': Basic gradient descent
        """
        self.layers = layers
        self.n_ensemble = layers[-1]
        self.key = jax.random.PRNGKey(seed)
        self.nn = _NN(layers, activation)
        self.loss_type = loss_type
        self.min_sigma = min_sigma
        self.optimizer = optimizer.lower()
        self.calibration_factor = 1.0  # Default: no calibration

    def fit(self, X, y, val_X=None, val_y=None, pretrain_with_mse=None, **kwargs):
        """Fit the DPOSE model with calibrated uncertainty estimation.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            val_X: Optional validation features for post-hoc calibration.
            val_y: Optional validation targets for post-hoc calibration.
            pretrain_with_mse: If True, pre-train with MSE before NLL training (default: auto).
                              - For 'nll': defaults to True (robust two-stage training)
                              - For 'crps'/'mse': defaults to False (not needed)
                              Set to False to disable pre-training for NLL (not recommended).
            **kwargs: Additional arguments passed to the optimizer. Common parameters:
                     - maxiter: Maximum iterations (default: 1500)
                     - tol: Convergence tolerance (default: 1e-3)
                     - pretrain_maxiter: Iterations for MSE pre-training (default: 500)

                     Optimizer-specific parameters:
                     - BFGS/LBFGS: stepsize, linesearch, max_linesearch_iter
                     - Adam: learning_rate (default: 1e-3), b1, b2, eps
                     - SGD: learning_rate, momentum
                     - Muon: learning_rate (default: 0.02), beta (default: 0.95),
                             ns_steps (default: 5), weight_decay
                     - See jaxopt documentation for full parameter lists

        Returns:
            self: Fitted model.
        """
        # Auto-detect if we should pre-train
        if pretrain_with_mse is None:
            pretrain_with_mse = self.loss_type == "nll"

        # Extract pre-training specific kwargs
        pretrain_maxiter = kwargs.pop("pretrain_maxiter", 500)

        # Stage 1: MSE pre-training (if using NLL and pretrain enabled)
        if pretrain_with_mse and self.loss_type == "nll":
            print("\n" + "=" * 70)
            print("NLL TRAINING: Two-Stage Approach for Robustness")
            print("=" * 70)
            print(f"Stage 1: MSE pre-training ({pretrain_maxiter} iterations)")
            print("         → Ensures good predictions before uncertainty calibration")

            # Temporarily switch to MSE
            original_loss = self.loss_type
            self.loss_type = "mse"

            # Create kwargs for pre-training
            pretrain_kwargs = kwargs.copy()
            pretrain_kwargs["maxiter"] = pretrain_maxiter

            # Pre-train with MSE
            self._fit_internal(X, y, val_X=None, val_y=None, **pretrain_kwargs)

            # Report pre-training results
            y_pred_pretrain = self.predict(X)
            mae_pretrain = np.mean(np.abs(y - y_pred_pretrain))
            print(f"         ✓ Pre-training complete: MAE = {mae_pretrain:.6f}")

            # Restore NLL
            self.loss_type = original_loss

            print("\nStage 2: NLL fine-tuning (uncertainty calibration)")
            print("         → Calibrating uncertainties while maintaining accuracy")
            print("=" * 70 + "\n")

        # Stage 2: Main training (NLL, CRPS, or MSE)
        return self._fit_internal(X, y, val_X, val_y, **kwargs)

    def _fit_internal(self, X, y, val_X=None, val_y=None, **kwargs):
        """Internal method for actual fitting (used by fit() for each stage).

        This is separated out to enable two-stage training for NLL.
        """
        # Initialize or reuse parameters
        if not hasattr(self, "optpars"):
            params = self.nn.init(self.key, X)  # Dummy input to init
        else:
            params = self.optpars

        @jit
        def objective(pars):
            """Loss function with per-sample uncertainty from ensemble spread."""
            # Get ensemble predictions: shape (n_samples, n_ensemble)
            pY = self.nn.apply(pars, np.asarray(X))

            # Ensemble statistics
            py = pY.mean(axis=1)  # Predicted mean (n_samples,)
            # Uncertainty with numerically stable gradient (avoids NaN when ensemble members are identical)
            sigma = np.sqrt(
                pY.var(axis=1) + self.min_sigma**2
            )  # Predicted uncertainty (n_samples,)

            # Prediction errors
            errs = np.asarray(y).ravel() - py

            if self.loss_type == "nll":
                # Negative Log-Likelihood (Kellner & Ceriotti, Eq. 6)
                # Penalizes both prediction errors AND miscalibrated uncertainties
                nll = 0.5 * (errs**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
                return np.mean(nll)

            elif self.loss_type == "crps":
                # Continuous Ranked Probability Score (Kellner & Ceriotti, Eq. 18)
                # More robust than NLL, less sensitive to outliers
                z = errs / sigma
                phi_z = jax.scipy.stats.norm.pdf(z)
                Phi_z = jax.scipy.stats.norm.cdf(z)
                crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))
                return np.mean(crps)

            elif self.loss_type == "mse":
                # Simple MSE (no uncertainty training)
                return np.mean(errs**2)

            else:
                raise ValueError(
                    f"Unknown loss_type: {self.loss_type}. Use 'nll', 'crps', or 'mse'."
                )

        # Solver configuration
        if "maxiter" not in kwargs:
            kwargs["maxiter"] = 1500
        if "tol" not in kwargs:
            kwargs["tol"] = 1e-3

        # Select optimizer
        if self.optimizer == "bfgs":
            solver = jaxopt.BFGS(fun=value_and_grad(objective), value_and_grad=True, **kwargs)
        elif self.optimizer == "lbfgs":
            solver = jaxopt.LBFGS(fun=value_and_grad(objective), value_and_grad=True, **kwargs)
        elif self.optimizer == "lbfgsb":
            solver = jaxopt.LBFGSB(fun=value_and_grad(objective), value_and_grad=True, **kwargs)
        elif self.optimizer == "nonlinear_cg":
            solver = jaxopt.NonlinearCG(
                fun=value_and_grad(objective), value_and_grad=True, **kwargs
            )
        elif self.optimizer == "adam":
            # Adam uses OptaxSolver with optax optimizer
            if "learning_rate" not in kwargs:
                kwargs["learning_rate"] = 1e-3
            solver = jaxopt.OptaxSolver(
                opt=optax.adam(kwargs.pop("learning_rate")), fun=objective, **kwargs
            )
        elif self.optimizer == "sgd":
            # SGD uses OptaxSolver with optax optimizer
            if "learning_rate" not in kwargs:
                kwargs["learning_rate"] = 1e-2
            lr = kwargs.pop("learning_rate")
            momentum = kwargs.pop("momentum", 0.9)
            solver = jaxopt.OptaxSolver(
                opt=optax.sgd(lr, momentum=momentum), fun=objective, **kwargs
            )
        elif self.optimizer == "muon":
            # Muon uses OptaxSolver with optax.contrib.muon
            # Muon orthogonalizes momentum updates for 2D parameters
            if "learning_rate" not in kwargs:
                kwargs["learning_rate"] = 0.02  # Muon typically uses higher LR than Adam
            lr = kwargs.pop("learning_rate")
            beta = kwargs.pop("beta", 0.95)
            ns_steps = kwargs.pop("ns_steps", 5)
            weight_decay = kwargs.pop("weight_decay", 0.0)

            solver = jaxopt.OptaxSolver(
                opt=optax.contrib.muon(
                    learning_rate=lr,
                    beta=beta,
                    ns_steps=ns_steps,
                    nesterov=True,
                    weight_decay=weight_decay,
                ),
                fun=objective,
                **kwargs,
            )
        elif self.optimizer == "gradient_descent":
            solver = jaxopt.GradientDescent(fun=objective, **kwargs)
        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer}. "
                f"Choose from: bfgs, lbfgs, lbfgsb, nonlinear_cg, adam, sgd, muon, gradient_descent"
            )

        # Optimize
        self.optpars, self.state = solver.run(params)

        # Post-hoc calibration on validation set if provided
        if val_X is not None and val_y is not None:
            self._calibrate(val_X, val_y)

        return self

    def _calibrate(self, X, y):
        """Apply post-hoc calibration using validation set.

        Implements Eq. 8 from Kellner & Ceriotti (2024):
        α² = (1/n_val) Σ [Δy(X)² / σ(X)²]

        This rescales uncertainties so that their magnitude matches
        actual prediction errors on the validation set.

        Args:
            X: Validation features.
            y: Validation targets.
        """
        pY = self.nn.apply(self.optpars, np.asarray(X))
        py = pY.mean(axis=1)
        sigma = np.sqrt(pY.var(axis=1) + self.min_sigma**2)

        errs = np.asarray(y).ravel() - py

        # Check for ensemble collapse
        mean_sigma = np.mean(sigma)
        if mean_sigma < 1e-8:
            print("\n⚠ WARNING: Ensemble has collapsed!")
            print(f"  Mean uncertainty: {mean_sigma:.2e} (nearly zero)")
            print(f"  Ensemble spread: {sigma.min():.2e} to {sigma.max():.2e}")
            print("\n  Possible causes:")
            print(f"    - Ensemble size too small (current: {self.n_ensemble})")
            print("    - Training with MSE loss (use 'nll' or 'crps')")
            print("    - Model overfit (reduce training iterations)")
            print("\n  Skipping calibration (using α = 1.0)")
            self.calibration_factor = 1.0
            return

        # Calibration factor: ratio of empirical to predicted variance
        alpha_sq = np.mean(errs**2) / np.mean(sigma**2)
        self.calibration_factor = float(np.sqrt(alpha_sq))

        # Check for numerical issues
        if not np.isfinite(self.calibration_factor):
            print(f"\n⚠ WARNING: Calibration failed (α = {self.calibration_factor})")
            print(f"  Mean error²: {np.mean(errs**2):.6f}")
            print(f"  Mean σ²: {np.mean(sigma**2):.6f}")
            print("  Skipping calibration (using α = 1.0)")
            self.calibration_factor = 1.0
            return

        print(f"\nCalibration factor α = {self.calibration_factor:.4f}")
        if self.calibration_factor > 1.5:
            print("  ⚠ Model is overconfident (α > 1.5)")
        elif self.calibration_factor < 0.7:
            print("  ⚠ Model is underconfident (α < 0.7)")
        else:
            print("  ✓ Model is well-calibrated")

    def report(self):
        """Print optimization diagnostics."""
        print("Optimization converged:")
        # Handle different state formats from different optimizers
        if hasattr(self.state, "iter_num"):
            print(f"  Iterations: {self.state.iter_num}")
        elif hasattr(self.state, "num_iter"):
            print(f"  Iterations: {self.state.num_iter}")

        if hasattr(self.state, "value"):
            print(f"  Final loss: {self.state.value:.6f}")

        print(f"  Optimizer: {self.optimizer}")
        print(f"  Ensemble size: {self.n_ensemble}")
        print(f"  Loss type: {self.loss_type}")
        if hasattr(self, "calibration_factor"):
            print(f"  Calibration: α = {self.calibration_factor:.4f}")

    def predict(self, X, return_std=False):
        """Make predictions with uncertainty estimates.

        Args:
            X: Input features, shape (n_samples, n_features).
            return_std: If True, return (predictions, uncertainties).

        Returns:
            predictions: Mean ensemble predictions, shape (n_samples,).
            uncertainties: Standard deviation (if return_std=True), shape (n_samples,).
        """
        X = np.atleast_2d(X)
        P = self.nn.apply(self.optpars, X)

        mean_pred = P.mean(axis=1)
        std_pred = np.sqrt(P.var(axis=1) + self.min_sigma**2)

        # Apply post-hoc calibration if available
        if hasattr(self, "calibration_factor") and self.calibration_factor != 1.0:
            std_pred = std_pred * self.calibration_factor

        if return_std:
            return mean_pred, std_pred
        else:
            return mean_pred

    def predict_ensemble(self, X):
        """Get full ensemble predictions for uncertainty propagation.

        This method is crucial for propagating uncertainties through
        non-linear transformations (Kellner & Ceriotti, Eq. 11).

        Example:
            # For some function f(y)
            ensemble_preds = model.predict_ensemble(X)  # (n_samples, n_ensemble)
            z_ensemble = f(ensemble_preds)              # Apply f to each member
            z_mean = z_ensemble.mean(axis=1)            # Mean of transformed quantity
            z_std = z_ensemble.std(axis=1)              # Uncertainty of transformed quantity

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            ensemble_predictions: Full ensemble output, shape (n_samples, n_ensemble).
        """
        X = np.atleast_2d(X)
        return self.nn.apply(self.optpars, X)

    def __call__(self, X, return_std=False, distribution=False):
        """Execute the model (alternative interface to predict).

        Args:
            X: Input features, shape (n_samples, n_features).
            return_std: If True, return uncertainties.
            distribution: If True, return full ensemble; else return mean.

        Returns:
            If distribution=False: predictions (and uncertainties if return_std=True).
            If distribution=True: full ensemble predictions, shape (n_samples, n_ensemble).
        """
        if not hasattr(self, "optpars"):
            raise Exception("You need to fit the model first.")

        X = np.atleast_2d(X)
        P = self.nn.apply(self.optpars, X)

        if distribution:
            # Return full ensemble
            if return_std:
                se = np.sqrt(P.var(axis=1) + self.min_sigma**2)
                if hasattr(self, "calibration_factor") and self.calibration_factor != 1.0:
                    se = se * self.calibration_factor
                return (P, se)
            else:
                return P
        else:
            # Return mean (and std if requested)
            mean_pred = P.mean(axis=1)
            if return_std:
                std_pred = np.sqrt(P.var(axis=1) + self.min_sigma**2)
                if hasattr(self, "calibration_factor") and self.calibration_factor != 1.0:
                    std_pred = std_pred * self.calibration_factor
                return (mean_pred, std_pred)
            else:
                return mean_pred

    def plot(self, X, y, distribution=False, ax=None):
        """Visualize predictions with uncertainty bands.

        Args:
            X: Input features, shape (n_samples, n_features).
               For 1D input, will be used as x-axis.
            y: True target values.
            distribution: If True, plot individual ensemble members.
            ax: Matplotlib axis (optional). If None, uses current axis.

        Returns:
            matplotlib figure object.
        """
        if ax is None:
            ax = plt.gca()

        # Get predictions with calibrated uncertainties
        mp, se = self.predict(X, return_std=True)

        # For line plots, need to sort by X
        X_plot = X.ravel()
        sort_idx = np.argsort(X_plot)
        X_sorted = X_plot[sort_idx]
        mp_sorted = mp[sort_idx]
        se_sorted = se[sort_idx]

        # Plot in correct z-order (back to front):

        # 1. Uncertainty band (background, lowest z-order)
        ax.fill_between(
            X_sorted,
            mp_sorted - 2 * se_sorted,
            mp_sorted + 2 * se_sorted,
            alpha=0.3,
            color="red",
            label="±2σ (95% CI)",
            zorder=1,
        )

        # 2. Individual ensemble members (if requested, middle layer)
        if distribution:
            P = self.nn.apply(self.optpars, X)
            P_sorted = P[sort_idx, :]
            # Plot all members at once (more efficient) with very low alpha
            ax.plot(X_sorted, P_sorted, "k-", alpha=0.05, linewidth=0.5, zorder=2)

        # 3. Mean prediction line (middle-front, should be visible)
        ax.plot(X_sorted, mp_sorted, "r-", label="mean prediction", linewidth=2.5, zorder=3)

        # 4. Data points (front, highest z-order so always visible)
        ax.plot(X_plot, y, "b.", label="data", alpha=0.7, markersize=8, zorder=4)

        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title(f"DPOSE Predictions (n_ensemble={self.n_ensemble})")
        ax.grid(True, alpha=0.3)

        return plt.gcf()

    def uncertainty_metrics(self, X, y):
        """Compute uncertainty quantification metrics.

        Following Kellner & Ceriotti (2024), computes several diagnostics
        to assess the quality of uncertainty estimates.

        Args:
            X: Input features.
            y: True target values.

        Returns:
            dict with keys:
                - 'rmse': Root mean squared error
                - 'mae': Mean absolute error
                - 'nll': Negative log-likelihood (lower is better)
                - 'miscalibration_area': Deviation from ideal calibration (lower is better)
                - 'z_score_mean': Should be ~0 if well-calibrated
                - 'z_score_std': Should be ~1 if well-calibrated
        """
        mp, se = self.predict(X, return_std=True)
        y = np.asarray(y).ravel()

        errs = y - mp
        rmse = np.sqrt(np.mean(errs**2))
        mae = np.mean(np.abs(errs))

        # Check for ensemble collapse (sigma too small)
        mean_se = np.mean(se)
        if mean_se < 1e-8:
            print("\n⚠ WARNING: Cannot compute uncertainty metrics - ensemble has collapsed!")
            print(f"  Mean uncertainty: {mean_se:.2e} (nearly zero)")
            print("  This causes division by zero in metric calculations.")
            print("\n  Returning basic accuracy metrics only (NLL, Z-scores unavailable)")

            return {
                "rmse": float(rmse),
                "mae": float(mae),
                "nll": float("nan"),
                "miscalibration_area": float("nan"),
                "z_score_mean": float("nan"),
                "z_score_std": float("nan"),
            }

        # Check for any numerical issues in uncertainties
        if not np.all(np.isfinite(se)) or np.any(se <= 0):
            print("\n⚠ WARNING: Invalid uncertainty values detected!")
            print(f"  Contains NaN: {np.any(np.isnan(se))}")
            print(f"  Contains inf: {np.any(np.isinf(se))}")
            print(f"  Contains zeros or negatives: {np.any(se <= 0)}")
            print("\n  Returning basic accuracy metrics only")

            return {
                "rmse": float(rmse),
                "mae": float(mae),
                "nll": float("nan"),
                "miscalibration_area": float("nan"),
                "z_score_mean": float("nan"),
                "z_score_std": float("nan"),
            }

        # NLL (Eq. 6)
        nll = 0.5 * np.mean(errs**2 / se**2 + np.log(2 * np.pi * se**2))

        # Standardized residuals (z-scores)
        z_scores = errs / se
        z_mean = np.mean(z_scores)
        z_std = np.std(z_scores)

        # Miscalibration area (Kellner Fig. 2)
        # Measures deviation of empirical CDF from theoretical Gaussian CDF
        sorted_z = np.sort(z_scores)
        empirical_cdf = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
        theoretical_cdf = jax.scipy.stats.norm.cdf(sorted_z)
        miscalibration_area = np.mean(np.abs(empirical_cdf - theoretical_cdf))

        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "nll": float(nll),
            "miscalibration_area": float(miscalibration_area),
            "z_score_mean": float(z_mean),
            "z_score_std": float(z_std),
        }

        return metrics

    def print_metrics(self, X, y):
        """Print uncertainty metrics in human-readable format.

        Args:
            X: Input features.
            y: True target values.
        """
        metrics = self.uncertainty_metrics(X, y)

        print("\n" + "=" * 50)
        print("UNCERTAINTY QUANTIFICATION METRICS")
        print("=" * 50)
        print("Prediction Accuracy:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")

        # Check if uncertainty metrics are available
        has_uncertainty = not np.isnan(metrics["nll"])

        if has_uncertainty:
            print("\nUncertainty Quality:")
            print(f"  NLL: {metrics['nll']:.6f} (lower is better)")
            print(f"  Miscalibration Area: {metrics['miscalibration_area']:.6f} (lower is better)")
            print("\nCalibration Diagnostics:")
            print(f"  Z-score mean: {metrics['z_score_mean']:.4f} (ideal: 0)")
            print(f"  Z-score std:  {metrics['z_score_std']:.4f} (ideal: 1)")

            # Interpret calibration
            if abs(metrics["z_score_mean"]) < 0.1 and abs(metrics["z_score_std"] - 1) < 0.2:
                print("  ✓ Well-calibrated uncertainties")
            elif metrics["z_score_std"] < 0.8:
                print("  ⚠ Overconfident (uncertainties too small)")
            elif metrics["z_score_std"] > 1.2:
                print("  ⚠ Underconfident (uncertainties too large)")
            else:
                print("  ⚠ Miscalibrated")
        else:
            print("\nUncertainty Quality:")
            print("  NLL: N/A (ensemble collapsed)")
            print("  Miscalibration Area: N/A")
            print("\nCalibration Diagnostics:")
            print("  Z-score mean: N/A")
            print("  Z-score std: N/A")
            print("\n  ✗ Uncertainty estimates not available due to ensemble collapse")
            print("  ➜ See warnings above for diagnostic information and suggested fixes")

        print("=" * 50 + "\n")
