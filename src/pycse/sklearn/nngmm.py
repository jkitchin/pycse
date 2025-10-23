"""Neural network with Gaussian Mixture Model regression.

Use a neural network as a nonlinear feature generator, then use a GMM
regression for the last layer to get uncertainty quantification.

The GMM approach can capture multimodal distributions and complex
uncertainty patterns, making it suitable for heteroscedastic noise.

Example:

    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from pycse.sklearn.nngmm import NeuralNetworkGMM

    # Generate data with heteroscedastic noise
    X = np.random.randn(200, 5)
    y = np.sum(X**2, axis=1) + (0.1 + 0.5*X[:, 0]**2) * np.random.randn(200)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Setup neural network
    nn = MLPRegressor(
        hidden_layer_sizes=(20, 200),
        activation='relu',
        solver='lbfgs',
        max_iter=1000
    )

    # Create and train NNGMM
    nngmm = NeuralNetworkGMM(nn, n_components=1)
    nngmm.fit(X_train, y_train, val_X=X_val, val_y=y_val)

    # Get predictions with uncertainty
    y_pred, y_std = nngmm.predict(X_val, return_std=True)

    # Visualize (for 1D input)
    nngmm.plot(X, y)

    # Print diagnostics
    nngmm.report()
    nngmm.print_metrics(X_val, y_val)

Requires: scikit-learn, numpy, matplotlib, gmr
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network._base import ACTIVATIONS
import numpy as np
import matplotlib.pyplot as plt
from gmr import GMM


class NeuralNetworkGMM(BaseEstimator, RegressorMixin):
    """sklearn-compatible neural network with GMM regression in last layer.

    The idea is you fit a neural network and replace the last linear layer
    with a Gaussian Mixture Model regressor to estimate uncertainty.

    The GMM can capture complex, multimodal uncertainty distributions.
    """

    def __init__(self, nn, n_components=1, n_samples=500):
        """Initialize the Neural Network GMM Regressor.

        Args:
            nn: An sklearn.neural_network.MLPRegressor instance
            n_components: Number of GMM components (default: 1)
            n_samples: Number of samples for uncertainty estimation (default: 500)
        """
        self.nn = nn
        self.n_components = n_components
        self.n_samples = n_samples
        self.calibration_factor = 1.0  # For post-hoc calibration

    def _feat(self, X):
        """Return neural network features for X.

        Extracts features from the last hidden layer of the neural network.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Features from last hidden layer, shape (n_samples, hidden_size)
        """
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
        from the output layer and uses those in the GMM regressor.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            val_X: Optional validation features for post-hoc calibration
            val_y: Optional validation targets for post-hoc calibration

        Returns:
            self: Fitted model
        """
        # Initial fit of neural network
        self.nn.fit(X, y)

        # Create GMM from features and targets
        self.gmm = GMM(n_components=self.n_components)
        features = self._feat(X)
        self.gmm.from_samples(np.hstack([features, y[:, None]]))

        # Post-hoc calibration on validation set
        if val_X is not None and val_y is not None:
            self._calibrate(val_X, val_y)

        return self

    def _calibrate(self, X, y):
        """Perform post-hoc calibration of uncertainties.

        Computes a calibration factor that rescales predicted uncertainties
        to better match empirical errors on the validation set.

        Args:
            X: Validation features
            y: Validation targets
        """
        # Get predictions and uncertainties
        y_pred, y_std = self.predict(X, return_std=True)
        errors = y - y_pred

        # Check for collapsed uncertainties
        mean_std = np.mean(y_std)
        if mean_std < 1e-8:
            print("\n⚠ WARNING: GMM has collapsed to deterministic predictions!")
            print(f"  Mean uncertainty: {mean_std:.2e} (nearly zero)")
            print(f"  Uncertainty spread: {y_std.min():.2e} to {y_std.max():.2e}")
            print("\n  Possible causes:")
            print(f"    - GMM components: {self.n_components} (try increasing)")
            print("    - Neural network overfit (reduce training iterations)")
            print("    - Too few samples for uncertainty estimation")
            print("\n  Skipping calibration (using α = 1.0)")
            self.calibration_factor = 1.0
            return

        # Calibration factor: ratio of empirical to predicted variance
        alpha_sq = np.mean(errors**2) / np.mean(y_std**2)
        self.calibration_factor = float(np.sqrt(alpha_sq))

        # Check for numerical issues
        if not np.isfinite(self.calibration_factor):
            print(f"\n⚠ WARNING: Calibration failed (α = {self.calibration_factor})")
            print(f"  Mean error²: {np.mean(errors**2):.6f}")
            print(f"  Mean σ²: {np.mean(y_std**2):.6f}")
            print("  Skipping calibration (using α = 1.0)")
            self.calibration_factor = 1.0
            return

        print(f"\nCalibration factor α = {self.calibration_factor:.4f}")
        if 0.9 <= self.calibration_factor <= 1.1:
            print("  ✓ Model is well-calibrated")

    def predict(self, X, return_std=False):
        """Predict output values for X.

        Args:
            X: Input features, shape (n_samples, n_features)
            return_std: If True, also return standard deviation for each prediction

        Returns:
            y_pred: Predicted values, shape (n_samples,)
            y_std: Standard deviations (if return_std=True), shape (n_samples,)
        """
        # Get features from neural network
        feat = self._feat(X)

        # GMM prediction indices (input dimensions)
        inds = np.arange(0, feat.shape[1])

        # Predict mean
        y = self.gmm.predict(inds, feat)

        if return_std:
            se = []
            for f in feat:
                # Condition GMM on the features
                g = self.gmm.condition(np.arange(len(f)), f)
                # Sample from conditional distribution
                samples = g.sample(self.n_samples)
                # Compute standard deviation
                se.append(np.std(samples))

            # Apply calibration factor
            return y, self.calibration_factor * np.array(se)
        else:
            return y

    def report(self):
        """Print model diagnostics and configuration."""
        print("\n" + "=" * 50)
        print("NEURAL NETWORK GMM MODEL")
        print("=" * 50)
        print("Neural Network:")
        print(f"  Architecture: {self.nn.hidden_layer_sizes}")
        print(f"  Activation: {self.nn.activation}")
        print(f"  Solver: {self.nn.solver}")
        print(f"  Iterations: {self.nn.n_iter_}")
        print("\nGMM Configuration:")
        print(f"  Components: {self.n_components}")
        print(f"  Samples for UQ: {self.n_samples}")
        print("\nCalibration:")
        print(f"  Calibration factor α: {self.calibration_factor:.4f}")
        print("=" * 50 + "\n")

    def plot(self, X, y, ax=None):
        """Plot predictions with uncertainty bands.

        Only works for 1D input (or shows first feature if multidimensional).

        Args:
            X: Input features
            y: True targets
            ax: Optional matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Get predictions
        y_pred, y_std = self.predict(X, return_std=True)

        # Handle multi-dimensional input (use first feature)
        if X.ndim > 1 and X.shape[1] > 1:
            X_plot = X[:, 0]
            print(f"Note: Plotting first feature only (input has {X.shape[1]} features)")
        else:
            X_plot = X.ravel()

        # Sort by X for smooth line plots
        sort_idx = np.argsort(X_plot)
        X_sorted = X_plot[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]

        # Plot uncertainty band (95% CI)
        ax.fill_between(
            X_sorted,
            y_pred_sorted - 2 * y_std_sorted,
            y_pred_sorted + 2 * y_std_sorted,
            alpha=0.3,
            color="red",
            label="±2σ (95% CI)",
            zorder=1,
        )

        # Plot mean prediction
        ax.plot(X_sorted, y_pred_sorted, "r-", linewidth=2.5, label="GMM prediction", zorder=3)

        # Plot data points
        ax.scatter(X_plot, y, alpha=0.5, s=30, color="blue", label="Data", zorder=4)

        ax.set_xlabel("X" if X.ndim == 1 else "X[0]", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(
            "Neural Network GMM: Predictions with Uncertainty", fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return ax

    def uncertainty_metrics(self, X, y):
        """Compute uncertainty quantification metrics.

        Evaluates how well the model's uncertainty estimates match
        the empirical errors.

        Args:
            X: Input features
            y: True targets

        Returns:
            dict: Dictionary containing metrics:
                - rmse: Root mean squared error
                - mae: Mean absolute error
                - nll: Negative log-likelihood
                - miscalibration_area: Area between calibration curve and ideal
                - z_score_mean: Mean of z-scores (should be ~0)
                - z_score_std: Std of z-scores (should be ~1)
        """
        y_pred, y_std = self.predict(X, return_std=True)
        errors = y - y_pred

        # Basic accuracy
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))

        # Check for collapsed uncertainties
        mean_se = np.mean(y_std)
        if mean_se < 1e-8:
            print("\n⚠ WARNING: Cannot compute uncertainty metrics - GMM has collapsed!")
            print(f"  Mean uncertainty: {mean_se:.2e} (nearly zero)")
            print("  This causes division by zero in metric calculations.")
            print("\n  Returning basic accuracy metrics only (NLL, Z-scores unavailable)")

            return {
                "rmse": rmse,
                "mae": mae,
                "nll": float("nan"),
                "miscalibration_area": float("nan"),
                "z_score_mean": float("nan"),
                "z_score_std": float("nan"),
            }

        # Check for any numerical issues
        if not np.all(np.isfinite(y_std)) or np.any(y_std <= 0):
            print("\n⚠ WARNING: Invalid uncertainty values detected!")
            print(f"  Contains NaN: {np.any(np.isnan(y_std))}")
            print(f"  Contains inf: {np.any(np.isinf(y_std))}")
            print(f"  Contains zeros or negatives: {np.any(y_std <= 0)}")
            print("\n  Returning basic accuracy metrics only")

            return {
                "rmse": rmse,
                "mae": mae,
                "nll": float("nan"),
                "miscalibration_area": float("nan"),
                "z_score_mean": float("nan"),
                "z_score_std": float("nan"),
            }

        # Negative log-likelihood (Gaussian)
        nll = float(0.5 * np.mean((errors / y_std) ** 2 + np.log(2 * np.pi * y_std**2)))

        # Z-scores (standardized residuals)
        z_scores = errors / y_std
        z_mean = float(np.mean(z_scores))
        z_std = float(np.std(z_scores))

        # Miscalibration area (empirical calibration curve)
        # Sort by predicted uncertainty
        sorted_indices = np.argsort(y_std)
        sorted_errors = np.abs(errors[sorted_indices])
        sorted_stds = y_std[sorted_indices]

        # Compute cumulative calibration
        n = len(sorted_errors)
        expected_coverage = np.linspace(0, 1, n)
        actual_coverage = np.array(
            [np.mean(sorted_errors <= k * sorted_stds) for k in np.linspace(0, 3, n)]
        )

        # Compute miscalibration area
        miscalibration_area = float(np.mean(np.abs(actual_coverage - expected_coverage)))

        return {
            "rmse": rmse,
            "mae": mae,
            "nll": nll,
            "miscalibration_area": miscalibration_area,
            "z_score_mean": z_mean,
            "z_score_std": z_std,
        }

    def print_metrics(self, X, y):
        """Print comprehensive uncertainty metrics.

        Args:
            X: Input features
            y: True targets
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
            print("  NLL: N/A (GMM collapsed)")
            print("  Miscalibration Area: N/A")
            print("\nCalibration Diagnostics:")
            print("  Z-score mean: N/A")
            print("  Z-score std: N/A")
            print("\n  ✗ Uncertainty estimates not available due to collapsed GMM")
            print("  ➜ Try increasing n_components or reducing neural network training")

        print("=" * 50 + "\n")
