"""Neural network with Bayesian Linear regression.

Use a neural network as a nonlinear feature generator, then use Bayesian Linear
regression for the last layer so you can also get UQ.

Example:

    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import train_test_split

    # Generate data
    X = np.random.randn(200, 5)
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(200)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Setup neural network
    nn = MLPRegressor(
        hidden_layer_sizes=(20, 200),
        activation='relu',
        solver='lbfgs',
        max_iter=1000
    )

    # Setup Bayesian Ridge
    br = BayesianRidge(
        tol=1e-6,
        fit_intercept=False,
        compute_score=True
    )

    # Create and train NNBR
    nnbr = NeuralNetworkBLR(nn, br)
    nnbr.fit(X_train, y_train, val_X=X_val, val_y=y_val)

    # Get predictions with uncertainty
    y_pred, y_std = nnbr.predict(X_val, return_std=True)

    # Visualize (for 1D input)
    nnbr.plot(X, y)

    # Print diagnostics
    nnbr.report()
    nnbr.print_metrics(X_val, y_val)

Requires: scikit-learn, numpy, matplotlib
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network._base import ACTIVATIONS
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetworkBLR(BaseEstimator, RegressorMixin):
    """sklearn-compatible neural network with Bayesian Regression in last layer.

    The idea is you fit a neural network and replace the last linear layer with
    a Bayesian linear regressor so you can estimate uncertainty.
    """

    def __init__(self, nn, br):
        """Initialize the Neural Network Bayesian Linear Regressor.

        Args:
            nn: An sklearn.neural_network.MLPRegressor instance
            br: An sklearn.linear_model.BayesianRidge instance
        """
        self.nn = nn
        self.br = br
        self.calibration_factor = 1.0  # For post-hoc calibration

    def _feat(self, X):
        """Return neural network features for X."""
        weights = self.nn.coefs_
        biases = self.nn.intercepts_

        # Get the output of last hidden layer
        feat = X @ weights[0] + biases[0]
        ACTIVATIONS[self.nn.activation](feat)  # works in place
        for i in range(1, len(weights) - 1):
            feat = feat @ weights[i] + biases[i]
            ACTIVATIONS[self.nn.activation](feat)
        return feat

    def fit(self, X, y, val_X=None, val_y=None):
        """Fit the regressor to X, y.

        This first fits the NeuralNetwork instance. Then it gets the features
        from the output layer and uses those in the Bayesian linear regressor.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            val_X: Optional validation features for post-hoc calibration
            val_y: Optional validation targets for post-hoc calibration

        Returns:
            self: Fitted model
        """
        # Stage 1: Fit neural network
        self.nn.fit(X, y)

        # Stage 2: Bayesian linear regression on features
        self.br.fit(self._feat(X), y)

        # Stage 3: Post-hoc calibration if validation data provided
        if val_X is not None and val_y is not None:
            self._calibrate(val_X, val_y)

        return self

    def _calibrate(self, X, y):
        """Apply post-hoc calibration using validation set.

        Rescales uncertainties so that their magnitude matches
        actual prediction errors on the validation set.

        Args:
            X: Validation features
            y: Validation targets
        """
        y_pred, y_std = self.predict(X, return_std=True)
        errs = np.asarray(y).ravel() - y_pred

        # Check for near-zero uncertainties
        mean_std = np.mean(y_std)
        if mean_std < 1e-8:
            print("\n⚠ WARNING: Uncertainties are near zero!")
            print(f"  Mean uncertainty: {mean_std:.2e}")
            print("  Skipping calibration (using α = 1.0)")
            self.calibration_factor = 1.0
            return

        # Calibration factor: ratio of empirical to predicted variance
        alpha_sq = np.mean(errs**2) / np.mean(y_std**2)
        self.calibration_factor = float(np.sqrt(alpha_sq))

        # Check for numerical issues
        if not np.isfinite(self.calibration_factor):
            print(f"\n⚠ WARNING: Calibration failed (α = {self.calibration_factor})")
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

    def predict(self, X, return_std=False):
        """Predict output values for X.

        Args:
            X: Input features, shape (n_samples, n_features)
            return_std: If True, return (predictions, uncertainties)

        Returns:
            predictions: Mean predictions, shape (n_samples,)
            uncertainties: Standard deviation (if return_std=True), shape (n_samples,)
        """
        result = self.br.predict(self._feat(X), return_std=return_std)

        if return_std:
            y_pred, y_std = result
            # Apply calibration if available
            if hasattr(self, "calibration_factor") and self.calibration_factor != 1.0:
                y_std = y_std * self.calibration_factor
            return y_pred, y_std
        else:
            return result

    def report(self):
        """Print model diagnostics."""
        print("Model Report:")
        print("  Neural Network:")
        print(f"    Architecture: {self.nn.hidden_layer_sizes}")
        print(f"    Activation: {self.nn.activation}")
        print(f"    Solver: {self.nn.solver}")
        print(f"    Iterations: {self.nn.n_iter_}")
        print(
            f"    Final loss: {self.nn.loss_:.6f}"
            if hasattr(self.nn, "loss_")
            else "    Final loss: N/A"
        )
        print("  Bayesian Ridge:")
        print(f"    Alpha (precision): {self.br.alpha_:.6f}")
        print(f"    Lambda (noise): {self.br.lambda_:.6f}")
        print(f"    Scores available: {len(self.br.scores_) if hasattr(self.br, 'scores_') else 0}")
        if hasattr(self, "calibration_factor"):
            print(f"  Calibration: α = {self.calibration_factor:.4f}")

    def plot(self, X, y, ax=None):
        """Visualize predictions with uncertainty bands.

        Args:
            X: Input features, shape (n_samples, n_features)
               For 1D input, will be used as x-axis
            y: True target values
            ax: Matplotlib axis (optional). If None, uses current axis

        Returns:
            matplotlib figure object
        """
        if ax is None:
            ax = plt.gca()

        # Get predictions with calibrated uncertainties
        y_pred, y_std = self.predict(X, return_std=True)

        # For line plots, need to sort by X
        X_plot = X.ravel()
        sort_idx = np.argsort(X_plot)
        X_sorted = X_plot[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]

        # Plot in correct z-order (back to front):

        # 1. Uncertainty band (background)
        ax.fill_between(
            X_sorted,
            y_pred_sorted - 2 * y_std_sorted,
            y_pred_sorted + 2 * y_std_sorted,
            alpha=0.3,
            color="red",
            label="±2σ (95% CI)",
            zorder=1,
        )

        # 2. Mean prediction line
        ax.plot(X_sorted, y_pred_sorted, "r-", label="mean prediction", linewidth=2.5, zorder=3)

        # 3. Data points (front)
        ax.plot(X_plot, y, "b.", label="data", alpha=0.7, markersize=8, zorder=4)

        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title(f"NNBR Predictions (NN: {self.nn.hidden_layer_sizes})")
        ax.grid(True, alpha=0.3)

        return plt.gcf()

    def uncertainty_metrics(self, X, y):
        """Compute uncertainty quantification metrics.

        Args:
            X: Input features
            y: True target values

        Returns:
            dict with keys:
                - 'rmse': Root mean squared error
                - 'mae': Mean absolute error
                - 'nll': Negative log-likelihood (lower is better)
                - 'miscalibration_area': Deviation from ideal calibration (lower is better)
                - 'z_score_mean': Should be ~0 if well-calibrated
                - 'z_score_std': Should be ~1 if well-calibrated
        """
        y_pred, y_std = self.predict(X, return_std=True)
        y = np.asarray(y).ravel()

        errs = y - y_pred
        rmse = np.sqrt(np.mean(errs**2))
        mae = np.mean(np.abs(errs))

        # Check for near-zero uncertainties
        mean_std = np.mean(y_std)
        if mean_std < 1e-8:
            print("\n⚠ WARNING: Cannot compute uncertainty metrics - uncertainties are near zero!")
            print(f"  Mean uncertainty: {mean_std:.2e}")
            return {
                "rmse": float(rmse),
                "mae": float(mae),
                "nll": float("nan"),
                "miscalibration_area": float("nan"),
                "z_score_mean": float("nan"),
                "z_score_std": float("nan"),
            }

        # Check for numerical issues
        if not np.all(np.isfinite(y_std)) or np.any(y_std <= 0):
            print("\n⚠ WARNING: Invalid uncertainty values detected!")
            return {
                "rmse": float(rmse),
                "mae": float(mae),
                "nll": float("nan"),
                "miscalibration_area": float("nan"),
                "z_score_mean": float("nan"),
                "z_score_std": float("nan"),
            }

        # NLL (negative log-likelihood)
        nll = 0.5 * np.mean(errs**2 / y_std**2 + np.log(2 * np.pi * y_std**2))

        # Standardized residuals (z-scores)
        z_scores = errs / y_std
        z_mean = np.mean(z_scores)
        z_std = np.std(z_scores)

        # Miscalibration area
        sorted_z = np.sort(z_scores)
        empirical_cdf = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
        # For theoretical CDF, use scipy if available, else simple approximation
        try:
            from scipy.stats import norm

            theoretical_cdf = norm.cdf(sorted_z)
        except ImportError:
            # Simple approximation using error function
            theoretical_cdf = 0.5 * (1 + np.tanh(sorted_z / np.sqrt(2)))

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
            X: Input features
            y: True target values
        """
        metrics = self.uncertainty_metrics(X, y)

        print("\n" + "=" * 50)
        print("UNCERTAINTY QUANTIFICATION METRICS (NNBR)")
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
            print("  NLL: N/A (uncertainties near zero)")
            print("\n  ✗ Uncertainty estimates not available")

        print("=" * 50 + "\n")
