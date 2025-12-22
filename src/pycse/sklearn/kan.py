"""A KAN (Kolmogorov-Arnold Network) model in JAX.

Implementation of Kolmogorov-Arnold Networks based on:
Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.

Key features:
- Learnable activation functions using B-splines on edges
- Replaces fixed activations on nodes with learnable functions on edges
- sklearn-compatible API with uncertainty quantification
- Post-hoc calibration on validation set

KANs are based on the Kolmogorov-Arnold representation theorem, which states
that any multivariate continuous function can be represented as a composition
of continuous functions of a single variable and addition.

Example usage:

    import numpy as np
    from pycse.sklearn.kan import KAN

    # Generate data
    X = np.linspace(0, 1, 100)[:, None]
    y = np.sin(2 * np.pi * X.ravel()) + 0.1 * np.random.randn(100)

    # Train KAN
    model = KAN(layers=(1, 5, 1), grid_size=5)
    model.fit(X, y)

    # Predict with uncertainty
    y_pred, y_std = model.predict(X, return_std=True)

    # Visualize
    model.plot(X, y)

Requires: flax, optax, jax, scikit-learn
"""

import os

import jax
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from flax import linen as nn

from pycse.sklearn.optimizers import run_optimizer

os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


def b_spline_basis(x, grid, k=3):
    """Compute B-spline basis functions using Cox-de Boor recursion.

    Args:
        x: Input values, shape (n_samples,).
        grid: Knot positions, shape (n_knots,).
        k: Spline order (default 3 for cubic splines).

    Returns:
        B-spline basis values, shape (n_samples, n_basis).
    """
    n_intervals = len(grid) - 1

    # Order 0 (step functions)
    bases = []
    for i in range(n_intervals):
        # B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1], else 0
        # Handle rightmost interval specially to include right endpoint
        if i == n_intervals - 1:
            b = jnp.where((x >= grid[i]) & (x <= grid[i + 1]), 1.0, 0.0)
        else:
            b = jnp.where((x >= grid[i]) & (x < grid[i + 1]), 1.0, 0.0)
        bases.append(b)

    bases = jnp.stack(bases, axis=-1)  # (n_samples, n_intervals)

    # Cox-de Boor recursion for higher orders
    for order in range(1, k + 1):
        new_bases = []
        n_basis = bases.shape[-1] - 1

        for i in range(n_basis):
            # Left term: (x - t_i) / (t_{i+order} - t_i) * B_{i,order-1}
            denom_left = grid[i + order] - grid[i]
            left = jnp.where(
                denom_left > 1e-10,
                (x - grid[i]) / denom_left * bases[..., i],
                0.0,
            )

            # Right term: (t_{i+order+1} - x) / (t_{i+order+1} - t_{i+1}) * B_{i+1,order-1}
            denom_right = grid[i + order + 1] - grid[i + 1]
            right = jnp.where(
                denom_right > 1e-10,
                (grid[i + order + 1] - x) / denom_right * bases[..., i + 1],
                0.0,
            )

            new_bases.append(left + right)

        if len(new_bases) > 0:
            bases = jnp.stack(new_bases, axis=-1)
        else:
            break

    return bases


class KANLayer(nn.Module):
    """A single KAN layer with learnable spline activations.

    Each edge has its own learnable activation function parameterized
    as a weighted sum of B-spline basis functions, plus a residual
    connection through a SiLU activation (as in the original paper).

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of grid intervals for B-splines.
        spline_order: Order of the B-spline (default 3 for cubic).
        grid_range: Range of the spline grid (min, max).
    """

    in_features: int
    out_features: int
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple = (-1.0, 1.0)

    def setup(self):
        """Initialize grid and parameters."""
        # Extended grid for B-splines (need extra knots for boundary conditions)
        n_knots = self.grid_size + 1 + 2 * self.spline_order
        self.grid = jnp.linspace(
            self.grid_range[0] - self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
            self.grid_range[1] + self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
            n_knots,
        )
        self.n_basis = self.grid_size + self.spline_order

    @nn.compact
    def __call__(self, x):
        """Forward pass through KAN layer.

        Args:
            x: Input tensor, shape (batch_size, in_features).

        Returns:
            Output tensor, shape (batch_size, out_features).
        """
        batch_size = x.shape[0]

        # Spline coefficients: shape (in_features, out_features, n_basis)
        # Initialize with small random values
        spline_weight = self.param(
            "spline_weight",
            nn.initializers.normal(stddev=0.1),
            (self.in_features, self.out_features, self.n_basis),
        )

        # Base weight for residual connection (like standard linear layer)
        base_weight = self.param(
            "base_weight",
            nn.initializers.xavier_uniform(),
            (self.in_features, self.out_features),
        )

        # Scale parameters for the spline and base contributions
        spline_scale = self.param(
            "spline_scale",
            nn.initializers.ones,
            (self.in_features, self.out_features),
        )

        # Normalize input to grid range for stable spline computation
        x_norm = jnp.clip(x, self.grid_range[0], self.grid_range[1])

        # Compute B-spline basis for each input feature
        # x_norm: (batch_size, in_features)
        output = jnp.zeros((batch_size, self.out_features))

        for i in range(self.in_features):
            # Get basis functions for this input dimension
            # basis: (batch_size, n_basis)
            basis = b_spline_basis(x_norm[:, i], self.grid, k=self.spline_order)

            for j in range(self.out_features):
                # Spline activation: weighted sum of basis functions
                # spline_out: (batch_size,)
                spline_out = jnp.dot(basis, spline_weight[i, j, :])

                # Base activation with SiLU (swish) as in the KAN paper
                base_out = nn.silu(x[:, i]) * base_weight[i, j]

                # Combine with learnable scale
                output = output.at[:, j].add(
                    spline_scale[i, j] * spline_out + base_out
                )

        return output


class _KANN(nn.Module):
    """A Kolmogorov-Arnold Network using learnable spline activations.

    Attributes:
        layers: Tuple of layer sizes (input_dim, hidden1, ..., output_dim).
        grid_size: Number of grid intervals for B-splines in each layer.
        spline_order: Order of B-splines (default 3 for cubic).
        grid_range: Input range for spline normalization.
    """

    layers: tuple
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple = (-1.0, 1.0)

    @nn.compact
    def __call__(self, x):
        """Forward pass through the KAN.

        Args:
            x: Input tensor, shape (batch_size, n_features).

        Returns:
            Output tensor, shape (batch_size, output_dim).
        """
        # Process through KAN layers
        for i in range(len(self.layers) - 1):
            in_dim = self.layers[i]
            out_dim = self.layers[i + 1]

            x = KANLayer(
                in_features=in_dim,
                out_features=out_dim,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                grid_range=self.grid_range,
            )(x)

        return x


class KAN(BaseEstimator, RegressorMixin):
    """KAN: Kolmogorov-Arnold Network regressor.

    A neural network that uses learnable activation functions (B-splines)
    on edges instead of fixed activations on nodes. Based on the
    Kolmogorov-Arnold representation theorem.

    Key Features:
    - Learnable spline activations for each edge
    - More interpretable than standard MLPs
    - Can represent complex functions with fewer parameters
    - Includes uncertainty quantification through ensemble output
    - Post-hoc calibration on validation set

    The last element of `layers` determines the ensemble size for UQ.
    For example, layers=(5, 10, 32) creates:
    - Input: 5 features
    - Hidden: 10 KAN neurons
    - Output: 32 ensemble members (for uncertainty quantification)

    For single-output regression without UQ, use layers=(n_features, ..., 1).
    """

    def __init__(
        self,
        layers,
        grid_size=5,
        spline_order=3,
        grid_range=(-2.0, 2.0),
        seed=19,
        optimizer="bfgs",
        loss_type="mse",
        min_sigma=1e-3,
    ):
        """Initialize a KAN model.

        Args:
            layers: Tuple of integers for neurons in each layer.
                   layers[0] is input dimension.
                   layers[-1] is output dimension (or ensemble size for UQ).
            grid_size: Number of grid intervals for B-splines (default: 5).
                      More intervals = more expressive but more parameters.
            spline_order: Order of B-spline (default: 3 for cubic).
                         Higher order = smoother but more computation.
            grid_range: Range for input normalization (default: (-2, 2)).
                       Should cover expected input range after normalization.
            seed: Random seed for weight initialization.
            optimizer: Optimization algorithm. Options:
                      - 'bfgs', 'lbfgs': L-BFGS (recommended)
                      - 'adam': Adam optimizer
                      - 'sgd': SGD with momentum
            loss_type: Loss function - 'mse' or 'crps' (default: 'mse').
            min_sigma: Minimum std dev for numerical stability (default: 1e-3).
        """
        self.layers = layers
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        self.optimizer = optimizer.lower()
        self.loss_type = loss_type
        self.min_sigma = min_sigma
        self.calibration_factor = 1.0
        self.n_ensemble = layers[-1]

        # Create the network
        self.nn = _KANN(
            layers=layers,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
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

    def fit(self, X, y, val_X=None, val_y=None, **kwargs):
        """Fit the KAN model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            val_X: Optional validation features for post-hoc calibration.
            val_y: Optional validation targets for post-hoc calibration.
            **kwargs: Additional arguments passed to the optimizer:
                     - maxiter: Maximum iterations (default: 1500)
                     - tol: Convergence tolerance (default: 1e-3)
                     - learning_rate: For Adam/SGD optimizers

        Returns:
            self: Fitted model.
        """
        X = jnp.atleast_2d(X)
        y = jnp.asarray(y).ravel()

        # Store normalization parameters
        self.X_mean_ = jnp.mean(X, axis=0)
        self.X_std_ = jnp.std(X, axis=0)
        self.y_mean_ = jnp.mean(y)
        self.y_std_ = jnp.std(y)

        # Normalize data
        X_norm = self._normalize_X(X)
        y_norm = self._normalize_y(y)

        # Initialize parameters
        params = self.nn.init(self.key, X_norm)

        @jit
        def objective(pars):
            """Loss function."""
            pY = self.nn.apply(pars, X_norm)

            if self.n_ensemble > 1:
                # Ensemble output for UQ
                py = pY.mean(axis=1)
                sigma = jnp.sqrt(pY.var(axis=1) + self.min_sigma**2)
                errs = y_norm - py

                if self.loss_type == "crps":
                    # CRPS loss for uncertainty training
                    z = errs / sigma
                    phi_z = jax.scipy.stats.norm.pdf(z)
                    Phi_z = jax.scipy.stats.norm.cdf(z)
                    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / jnp.sqrt(jnp.pi))
                    return jnp.mean(crps)
                else:
                    # MSE loss
                    return jnp.mean(errs**2)
            else:
                # Single output
                py = pY.ravel()
                errs = y_norm - py
                return jnp.mean(errs**2)

        # Run optimization
        maxiter = kwargs.pop("maxiter", 1500)
        tol = kwargs.pop("tol", 1e-3)

        self.optpars, self.state = run_optimizer(
            self.optimizer, objective, params, maxiter=maxiter, tol=tol, **kwargs
        )

        # Post-hoc calibration on validation set
        if val_X is not None and val_y is not None:
            self._calibrate(val_X, val_y)

        return self

    def _calibrate(self, X, y):
        """Apply post-hoc calibration using validation set.

        Args:
            X: Validation features.
            y: Validation targets.
        """
        if self.n_ensemble <= 1:
            # No calibration for single output
            return

        X = jnp.atleast_2d(X)
        y = jnp.asarray(y).ravel()

        X_norm = self._normalize_X(X)
        pY = self.nn.apply(self.optpars, X_norm)

        # Get predictions in normalized space
        py = pY.mean(axis=1)
        sigma = jnp.sqrt(pY.var(axis=1) + self.min_sigma**2)

        y_norm = self._normalize_y(y)
        errs = y_norm - py

        # Calibration factor
        mean_sigma = jnp.mean(sigma)
        if mean_sigma < 1e-8:
            print("\n⚠ WARNING: Ensemble has collapsed!")
            self.calibration_factor = 1.0
            return

        alpha_sq = jnp.mean(errs**2) / jnp.mean(sigma**2)
        self.calibration_factor = float(jnp.sqrt(alpha_sq))

        if not jnp.isfinite(self.calibration_factor):
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
        """Make predictions with optional uncertainty estimates.

        Args:
            X: Input features, shape (n_samples, n_features).
            return_std: If True, return (predictions, uncertainties).

        Returns:
            predictions: Mean predictions, shape (n_samples,).
            uncertainties: Standard deviation (if return_std=True).
        """
        X = jnp.atleast_2d(X)
        X_norm = self._normalize_X(X)
        pY = self.nn.apply(self.optpars, X_norm)

        if self.n_ensemble > 1:
            # Ensemble predictions
            mean_pred_norm = pY.mean(axis=1)
            std_pred_norm = jnp.sqrt(pY.var(axis=1) + self.min_sigma**2)

            # Denormalize
            mean_pred = self._denormalize_y(mean_pred_norm)
            std_pred = self._denormalize_std(std_pred_norm)

            # Apply calibration
            if self.calibration_factor != 1.0:
                std_pred = std_pred * self.calibration_factor

            if return_std:
                return mean_pred, std_pred
            return mean_pred
        else:
            # Single output
            pred_norm = pY.ravel()
            pred = self._denormalize_y(pred_norm)

            if return_std:
                # No uncertainty for single output
                return pred, jnp.zeros_like(pred)
            return pred

    def predict_ensemble(self, X):
        """Get full ensemble predictions for uncertainty propagation.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            ensemble_predictions: Full ensemble, shape (n_samples, n_ensemble).
        """
        X = jnp.atleast_2d(X)
        X_norm = self._normalize_X(X)
        pY_norm = self.nn.apply(self.optpars, X_norm)

        # Denormalize each ensemble member
        return self._denormalize_y(pY_norm)

    def report(self):
        """Print optimization diagnostics."""
        print("KAN Optimization Report:")
        print(f"  Architecture: {self.layers}")
        print(f"  Grid size: {self.grid_size}")
        print(f"  Spline order: {self.spline_order}")
        print(f"  Optimizer: {self.optimizer}")

        if hasattr(self.state, "iter_num"):
            print(f"  Iterations: {self.state.iter_num}")
        if hasattr(self.state, "value"):
            print(f"  Final loss: {self.state.value:.6f}")
        if hasattr(self.state, "converged"):
            print(f"  Converged: {self.state.converged}")
        if hasattr(self, "calibration_factor") and self.n_ensemble > 1:
            print(f"  Calibration: α = {self.calibration_factor:.4f}")

    def plot(self, X, y, ax=None, distribution=False):
        """Visualize predictions with uncertainty bands.

        Args:
            X: Input features (works best with 1D input).
            y: True target values.
            ax: Matplotlib axis (optional).
            distribution: If True, plot individual ensemble members.

        Returns:
            matplotlib figure object.
        """
        if ax is None:
            ax = plt.gca()

        X = np.asarray(X)
        y = np.asarray(y)

        # Get predictions
        mp, se = self.predict(X, return_std=True)
        mp = np.asarray(mp)
        se = np.asarray(se)

        # For 1D input, sort for line plot
        X_plot = X.ravel()
        sort_idx = np.argsort(X_plot)
        X_sorted = X_plot[sort_idx]
        mp_sorted = mp[sort_idx]
        se_sorted = se[sort_idx]

        # Plot uncertainty band
        if self.n_ensemble > 1:
            ax.fill_between(
                X_sorted,
                mp_sorted - 2 * se_sorted,
                mp_sorted + 2 * se_sorted,
                alpha=0.3,
                color="red",
                label="±2σ (95% CI)",
                zorder=1,
            )

        # Plot ensemble members
        if distribution and self.n_ensemble > 1:
            P = np.asarray(self.predict_ensemble(X))
            P_sorted = P[sort_idx, :]
            ax.plot(X_sorted, P_sorted, "k-", alpha=0.05, linewidth=0.5, zorder=2)

        # Plot mean prediction
        ax.plot(X_sorted, mp_sorted, "r-", label="KAN prediction", linewidth=2.5, zorder=3)

        # Plot data
        ax.plot(X_plot, y, "b.", label="data", alpha=0.7, markersize=8, zorder=4)

        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title(f"KAN Predictions (grid={self.grid_size}, layers={self.layers})")
        ax.grid(True, alpha=0.3)

        return plt.gcf()

    def uncertainty_metrics(self, X, y):
        """Compute uncertainty quantification metrics.

        Args:
            X: Input features.
            y: True target values.

        Returns:
            dict with keys: rmse, mae, nll, miscalibration_area, z_score_mean, z_score_std
        """
        mp, se = self.predict(X, return_std=True)
        y = jnp.asarray(y).ravel()
        mp = jnp.asarray(mp)
        se = jnp.asarray(se)

        errs = y - mp
        rmse = float(jnp.sqrt(jnp.mean(errs**2)))
        mae = float(jnp.mean(jnp.abs(errs)))

        if self.n_ensemble <= 1 or jnp.mean(se) < 1e-8:
            return {
                "rmse": rmse,
                "mae": mae,
                "nll": float("nan"),
                "miscalibration_area": float("nan"),
                "z_score_mean": float("nan"),
                "z_score_std": float("nan"),
            }

        # NLL
        nll = 0.5 * jnp.mean(errs**2 / se**2 + jnp.log(2 * jnp.pi * se**2))

        # Z-scores
        z_scores = errs / se
        z_mean = jnp.mean(z_scores)
        z_std = jnp.std(z_scores)

        # Miscalibration area
        sorted_z = jnp.sort(z_scores)
        empirical_cdf = jnp.arange(1, len(sorted_z) + 1) / len(sorted_z)
        theoretical_cdf = jax.scipy.stats.norm.cdf(sorted_z)
        miscalibration_area = jnp.mean(jnp.abs(empirical_cdf - theoretical_cdf))

        return {
            "rmse": rmse,
            "mae": mae,
            "nll": float(nll),
            "miscalibration_area": float(miscalibration_area),
            "z_score_mean": float(z_mean),
            "z_score_std": float(z_std),
        }

    def print_metrics(self, X, y):
        """Print uncertainty metrics in human-readable format.

        Args:
            X: Input features.
            y: True target values.
        """
        metrics = self.uncertainty_metrics(X, y)

        print("\n" + "=" * 50)
        print("KAN UNCERTAINTY QUANTIFICATION METRICS")
        print("=" * 50)
        print("Prediction Accuracy:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")

        if not np.isnan(metrics["nll"]):
            print("\nUncertainty Quality:")
            print(f"  NLL: {metrics['nll']:.6f}")
            print(f"  Miscalibration Area: {metrics['miscalibration_area']:.6f}")
            print("\nCalibration Diagnostics:")
            print(f"  Z-score mean: {metrics['z_score_mean']:.4f} (ideal: 0)")
            print(f"  Z-score std:  {metrics['z_score_std']:.4f} (ideal: 1)")
        else:
            print("\nUncertainty Quality: N/A (single output or collapsed ensemble)")

        print("=" * 50 + "\n")

    def score(self, X, y):
        """Return R² score on the given data.

        Args:
            X: Input features.
            y: True target values.

        Returns:
            R² score.
        """
        y_pred = self.predict(X)
        y = jnp.asarray(y).ravel()

        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)

        return float(1 - ss_res / ss_tot)

    def __call__(self, X, return_std=False, distribution=False):
        """Execute the model (alternative interface to predict).

        Args:
            X: Input features.
            return_std: If True, return uncertainties.
            distribution: If True, return full ensemble.

        Returns:
            Predictions (and uncertainties if requested).
        """
        if not hasattr(self, "optpars"):
            raise Exception("You need to fit the model first.")

        if distribution:
            return self.predict_ensemble(X)

        return self.predict(X, return_std=return_std)

    def plot_activations(self, layer_idx=0, figsize=(12, 8)):
        """Visualize the learned spline activation functions.

        This method plots the learned activation functions for each edge
        in a specified layer, showing how the B-spline activations have
        been shaped during training.

        Args:
            layer_idx: Which layer's activations to visualize (default: 0, first layer).
            figsize: Figure size tuple (width, height).

        Returns:
            matplotlib figure object.
        """
        if not hasattr(self, "optpars"):
            raise Exception("You need to fit the model first.")

        # Get parameters for the specified layer
        params = self.optpars["params"]
        layer_keys = [k for k in params.keys() if k.startswith("KANLayer_")]

        if layer_idx >= len(layer_keys):
            raise ValueError(f"layer_idx {layer_idx} out of range. Model has {len(layer_keys)} layers.")

        layer_key = layer_keys[layer_idx]
        layer_params = params[layer_key]

        spline_weight = np.asarray(layer_params["spline_weight"])
        base_weight = np.asarray(layer_params["base_weight"])
        spline_scale = np.asarray(layer_params["spline_scale"])

        in_features, out_features, n_basis = spline_weight.shape

        # Create grid for plotting
        n_knots = self.grid_size + 1 + 2 * self.spline_order
        grid = np.linspace(
            self.grid_range[0] - self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
            self.grid_range[1] + self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
            n_knots,
        )

        x_plot = np.linspace(self.grid_range[0], self.grid_range[1], 200)

        # Compute basis functions
        basis = np.asarray(b_spline_basis(jnp.array(x_plot), jnp.array(grid), k=self.spline_order))

        # Create subplots
        n_plots = min(in_features * out_features, 16)  # Limit to 16 subplots
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        plot_idx = 0
        for i in range(in_features):
            for j in range(out_features):
                if plot_idx >= n_plots:
                    break

                ax = axes[plot_idx]

                # Compute spline activation
                spline_out = np.dot(basis, spline_weight[i, j, :])

                # Compute base activation (SiLU)
                silu_out = x_plot / (1 + np.exp(-x_plot))  # SiLU = x * sigmoid(x)
                base_out = silu_out * base_weight[i, j]

                # Combined output
                combined = spline_scale[i, j] * spline_out + base_out

                # Plot
                ax.plot(x_plot, combined, 'b-', linewidth=2, label='Combined')
                ax.plot(x_plot, spline_scale[i, j] * spline_out, 'r--', linewidth=1, alpha=0.7, label='Spline')
                ax.plot(x_plot, base_out, 'g:', linewidth=1, alpha=0.7, label='Base (SiLU)')
                ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
                ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
                ax.set_title(f'Edge ({i}→{j})', fontsize=10)
                ax.grid(True, alpha=0.3)

                if plot_idx == 0:
                    ax.legend(fontsize=8)

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'Learned Activations - Layer {layer_idx}', fontsize=12)
        plt.tight_layout()
        return fig
