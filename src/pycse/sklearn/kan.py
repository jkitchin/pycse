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
    connection through a configurable base activation.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of grid intervals for B-splines.
        spline_order: Order of the B-spline (default 3 for cubic).
        grid_range: Range of the spline grid (min, max).
        base_activation: Activation for residual connection ('silu' or 'linear').
            Use 'linear' for exact MIP representation with spline_order=1.
    """

    in_features: int
    out_features: int
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple = (-1.0, 1.0)
    base_activation: str = "silu"

    def setup(self):
        """Initialize grid and parameters."""
        # Extended grid for B-splines (need extra knots for boundary conditions)
        n_knots = self.grid_size + 1 + 2 * self.spline_order
        self.grid = jnp.linspace(
            self.grid_range[0]
            - self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
            self.grid_range[1]
            + self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
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

                # Base activation (SiLU for expressiveness, linear for MIP)
                if self.base_activation == "silu":
                    base_out = nn.silu(x[:, i]) * base_weight[i, j]
                else:  # linear
                    base_out = x[:, i] * base_weight[i, j]

                # Combine with learnable scale
                output = output.at[:, j].add(spline_scale[i, j] * spline_out + base_out)

        return output


class _KANN(nn.Module):
    """A Kolmogorov-Arnold Network using learnable spline activations.

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
                base_activation=self.base_activation,
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
    - Supports multi-output regression
    - Includes uncertainty quantification through ensemble output
    - Post-hoc calibration on validation set

    Architecture is specified via `layers` for network structure and
    `n_ensemble` for uncertainty quantification:
    - layers=(5, 10, 2) with n_ensemble=1: 5 inputs → 10 hidden → 2 outputs (no UQ)
    - layers=(5, 10, 1) with n_ensemble=32: 5 inputs → 10 hidden → 1 output with 32 ensemble members (UQ)
    - layers=(5, 10, 3) with n_ensemble=10: 5 inputs → 10 hidden → 3 outputs, each with 10 ensemble members
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
        l1_spline=0.0,
        l1_activation=0.0,
        entropy_reg=0.0,
        base_activation="silu",
        n_ensemble=1,
    ):
        """Initialize a KAN model.

        Args:
            layers: Tuple of integers for neurons in each layer.
                   layers[0] is input dimension.
                   layers[-1] is output dimension (number of targets).
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
            l1_spline: L1 regularization on spline coefficients (default: 0.0).
                      Encourages sparse spline weights, simplifying activations.
            l1_activation: L1 regularization on activation outputs (default: 0.0).
                          Encourages sparse activation patterns.
            entropy_reg: Entropy regularization strength (default: 0.0).
                        Encourages activation functions to be more "decisive"
                        (closer to step functions), improving interpretability.
            base_activation: Base residual activation ('silu' or 'linear').
                           Use 'linear' with spline_order=1 for exact MIP export.
            n_ensemble: Number of ensemble members per output for UQ (default: 1).
                       Set > 1 to enable uncertainty quantification.
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
        self.l1_spline = l1_spline
        self.l1_activation = l1_activation
        self.entropy_reg = entropy_reg
        self.base_activation = base_activation
        self.n_ensemble = n_ensemble
        self.calibration_factor = 1.0
        self.n_outputs = layers[-1]

        # Internal network outputs n_outputs * n_ensemble
        internal_layers = layers[:-1] + (layers[-1] * n_ensemble,)

        # Create the network
        self.nn = _KANN(
            layers=internal_layers,
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
        """Normalize target values (handles multi-output)."""
        if self.y_mean_ is not None and self.y_std_ is not None:
            return (y - self.y_mean_) / (self.y_std_ + 1e-8)
        return y

    def _denormalize_y(self, y):
        """Denormalize predictions (handles multi-output and ensemble)."""
        if self.y_mean_ is not None and self.y_std_ is not None:
            # Handle 3D case: (batch, n_outputs, n_ensemble)
            if y.ndim == 3:
                # Reshape stats to (1, n_outputs, 1) for broadcasting
                y_std = self.y_std_.reshape(1, -1, 1)
                y_mean = self.y_mean_.reshape(1, -1, 1)
                return y * y_std + y_mean
            return y * self.y_std_ + self.y_mean_
        return y

    def _denormalize_std(self, std):
        """Denormalize standard deviations (handles multi-output)."""
        if self.y_std_ is not None:
            return std * self.y_std_
        return std

    def fit(self, X, y, val_X=None, val_y=None, **kwargs):
        """Fit the KAN model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,) or (n_samples, n_outputs).
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
        y = jnp.asarray(y)

        # Handle multi-output targets
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.shape[1] != self.n_outputs:
            raise ValueError(
                f"Target has {y.shape[1]} outputs but model expects {self.n_outputs}. "
                f"Set layers[-1]={y.shape[1]} to match."
            )

        # Store normalization parameters (per output)
        self.X_mean_ = jnp.mean(X, axis=0)
        self.X_std_ = jnp.std(X, axis=0)
        self.y_mean_ = jnp.mean(y, axis=0)  # Shape: (n_outputs,)
        self.y_std_ = jnp.std(y, axis=0)  # Shape: (n_outputs,)

        # Normalize data
        X_norm = self._normalize_X(X)
        y_norm = self._normalize_y(y)

        # Initialize parameters
        params = self.nn.init(self.key, X_norm)

        # Store regularization params for closure
        l1_spline = self.l1_spline
        l1_activation = self.l1_activation
        entropy_reg = self.entropy_reg
        n_outputs = self.n_outputs
        n_ensemble = self.n_ensemble
        min_sigma = self.min_sigma

        @jit
        def objective(pars):
            """Loss function with optional regularization."""
            pY_raw = self.nn.apply(pars, X_norm)
            # Reshape: (batch, n_outputs * n_ensemble) -> (batch, n_outputs, n_ensemble)
            pY = pY_raw.reshape(-1, n_outputs, n_ensemble)

            # Compute main loss
            if n_ensemble > 1:
                # Ensemble output for UQ: mean over ensemble dimension
                py = pY.mean(axis=2)  # (batch, n_outputs)
                sigma = jnp.sqrt(pY.var(axis=2) + min_sigma**2)  # (batch, n_outputs)
                errs = y_norm - py

                if self.loss_type == "crps":
                    # CRPS loss for uncertainty training
                    z = errs / sigma
                    phi_z = jax.scipy.stats.norm.pdf(z)
                    Phi_z = jax.scipy.stats.norm.cdf(z)
                    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / jnp.sqrt(jnp.pi))
                    loss = jnp.mean(crps)
                else:
                    # MSE loss
                    loss = jnp.mean(errs**2)
            else:
                # No ensemble: squeeze ensemble dimension
                py = pY.squeeze(axis=2)  # (batch, n_outputs)
                errs = y_norm - py
                loss = jnp.mean(errs**2)

            # Add regularization terms
            reg_loss = 0.0

            if l1_spline > 0 or entropy_reg > 0:
                # Iterate over layer parameters
                for key in pars["params"]:
                    if key.startswith("KANLayer_"):
                        layer_params = pars["params"][key]

                        if l1_spline > 0 and "spline_weight" in layer_params:
                            # L1 on spline coefficients
                            spline_w = layer_params["spline_weight"]
                            reg_loss = reg_loss + l1_spline * jnp.mean(jnp.abs(spline_w))

                        if entropy_reg > 0 and "spline_weight" in layer_params:
                            # Entropy regularization on spline weights
                            # Encourages weights to be more concentrated (less uniform)
                            spline_w = layer_params["spline_weight"]
                            # Normalize to get "probabilities" per edge
                            w_abs = jnp.abs(spline_w) + 1e-8
                            w_norm = w_abs / jnp.sum(w_abs, axis=-1, keepdims=True)
                            # Negative entropy (minimize to make weights more peaked)
                            entropy = -jnp.sum(w_norm * jnp.log(w_norm), axis=-1)
                            reg_loss = reg_loss + entropy_reg * jnp.mean(entropy)

            if l1_activation > 0:
                # L1 on activation outputs (encourages sparse activations)
                reg_loss = reg_loss + l1_activation * jnp.mean(jnp.abs(pY))

            return loss + reg_loss

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
            # No calibration without ensemble
            return

        X = jnp.atleast_2d(X)
        y = jnp.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_norm = self._normalize_X(X)
        pY_raw = self.nn.apply(self.optpars, X_norm)

        # Reshape: (batch, n_outputs * n_ensemble) -> (batch, n_outputs, n_ensemble)
        pY = pY_raw.reshape(-1, self.n_outputs, self.n_ensemble)

        # Get predictions in normalized space
        py = pY.mean(axis=2)  # (batch, n_outputs)
        sigma = jnp.sqrt(pY.var(axis=2) + self.min_sigma**2)

        y_norm = self._normalize_y(y)
        errs = y_norm - py

        # Calibration factor (averaged over all outputs)
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
            predictions: Mean predictions.
                - Shape (n_samples,) for single output (n_outputs=1)
                - Shape (n_samples, n_outputs) for multi-output
            uncertainties: Standard deviation per output (if return_std=True).
        """
        X = jnp.atleast_2d(X)
        X_norm = self._normalize_X(X)
        pY_raw = self.nn.apply(self.optpars, X_norm)

        # Reshape: (batch, n_outputs * n_ensemble) -> (batch, n_outputs, n_ensemble)
        pY = pY_raw.reshape(-1, self.n_outputs, self.n_ensemble)

        if self.n_ensemble > 1:
            # Ensemble predictions: mean over ensemble
            mean_pred_norm = pY.mean(axis=2)  # (batch, n_outputs)
            std_pred_norm = jnp.sqrt(pY.var(axis=2) + self.min_sigma**2)

            # Denormalize
            mean_pred = self._denormalize_y(mean_pred_norm)
            std_pred = self._denormalize_std(std_pred_norm)

            # Apply calibration
            if self.calibration_factor != 1.0:
                std_pred = std_pred * self.calibration_factor

            # Squeeze if single output
            if self.n_outputs == 1:
                mean_pred = mean_pred.squeeze(axis=1)
                std_pred = std_pred.squeeze(axis=1)

            if return_std:
                return mean_pred, std_pred
            return mean_pred
        else:
            # No ensemble: squeeze ensemble dimension
            pred_norm = pY.squeeze(axis=2)  # (batch, n_outputs)
            pred = self._denormalize_y(pred_norm)

            # Squeeze if single output
            if self.n_outputs == 1:
                pred = pred.squeeze(axis=1)

            if return_std:
                # No uncertainty without ensemble
                return pred, jnp.zeros_like(pred)
            return pred

    def predict_ensemble(self, X):
        """Get full ensemble predictions for uncertainty propagation.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            ensemble_predictions: Full ensemble.
                - Shape (n_samples, n_ensemble) for single output
                - Shape (n_samples, n_outputs, n_ensemble) for multi-output
        """
        X = jnp.atleast_2d(X)
        X_norm = self._normalize_X(X)
        pY_raw = self.nn.apply(self.optpars, X_norm)

        # Reshape: (batch, n_outputs * n_ensemble) -> (batch, n_outputs, n_ensemble)
        pY_norm = pY_raw.reshape(-1, self.n_outputs, self.n_ensemble)

        # Denormalize each output
        pY = self._denormalize_y(pY_norm)

        # Squeeze if single output
        if self.n_outputs == 1:
            pY = pY.squeeze(axis=1)  # (batch, n_ensemble)

        return pY

    def report(self):
        """Print optimization diagnostics."""
        print("KAN Optimization Report:")
        print(f"  Architecture: {self.layers}")
        print(f"  Grid size: {self.grid_size}")
        print(f"  Spline order: {self.spline_order}")
        print(f"  Optimizer: {self.optimizer}")

        # Show regularization if any is active
        if self.l1_spline > 0 or self.l1_activation > 0 or self.entropy_reg > 0:
            print("  Regularization:")
            if self.l1_spline > 0:
                print(f"    L1 spline: {self.l1_spline}")
            if self.l1_activation > 0:
                print(f"    L1 activation: {self.l1_activation}")
            if self.entropy_reg > 0:
                print(f"    Entropy: {self.entropy_reg}")

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
            raise ValueError(
                f"layer_idx {layer_idx} out of range. Model has {len(layer_keys)} layers."
            )

        layer_key = layer_keys[layer_idx]
        layer_params = params[layer_key]

        spline_weight = np.asarray(layer_params["spline_weight"])
        base_weight = np.asarray(layer_params["base_weight"])
        spline_scale = np.asarray(layer_params["spline_scale"])

        in_features, out_features, n_basis = spline_weight.shape

        # Create grid for plotting
        n_knots = self.grid_size + 1 + 2 * self.spline_order
        grid = np.linspace(
            self.grid_range[0]
            - self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
            self.grid_range[1]
            + self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
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
                ax.plot(x_plot, combined, "b-", linewidth=2, label="Combined")
                ax.plot(
                    x_plot,
                    spline_scale[i, j] * spline_out,
                    "r--",
                    linewidth=1,
                    alpha=0.7,
                    label="Spline",
                )
                ax.plot(x_plot, base_out, "g:", linewidth=1, alpha=0.7, label="Base (SiLU)")
                ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
                ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
                ax.set_title(f"Edge ({i}→{j})", fontsize=10)
                ax.grid(True, alpha=0.3)

                if plot_idx == 0:
                    ax.legend(fontsize=8)

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Learned Activations - Layer {layer_idx}", fontsize=12)
        plt.tight_layout()
        return fig

    def plot_network(self, figsize=None, edge_samples=50):
        """Visualize the KAN as a network graph with learned activations on edges.

        This creates a visualization similar to the original KAN paper, showing
        the network structure with the learned activation function shapes drawn
        on each edge.

        Args:
            figsize: Figure size tuple (width, height). If None, auto-computed.
            edge_samples: Number of points to sample for drawing activation curves.

        Returns:
            matplotlib figure object.
        """
        if not hasattr(self, "optpars"):
            raise Exception("You need to fit the model first.")

        params = self.optpars["params"]
        layer_keys = sorted([k for k in params.keys() if k.startswith("KANLayer_")])
        n_layers = len(layer_keys) + 1  # +1 for input layer

        # Build layer sizes from architecture
        layer_sizes = list(self.layers[:-1]) + [self.layers[-1] * self.n_ensemble]

        # Auto-compute figure size based on network dimensions
        max_nodes = max(layer_sizes)
        if figsize is None:
            figsize = (3 * n_layers, 2 * max_nodes)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")

        # Node positions
        node_positions = {}  # (layer, node_idx) -> (x, y)
        layer_spacing = 1.0
        node_spacing = 0.5

        for layer_idx, n_nodes in enumerate(layer_sizes):
            x = layer_idx * layer_spacing
            # Center nodes vertically
            y_offset = (max_nodes - n_nodes) * node_spacing / 2
            for node_idx in range(n_nodes):
                y = y_offset + node_idx * node_spacing
                node_positions[(layer_idx, node_idx)] = (x, y)

        # Precompute activation curves for all edges
        x_plot = np.linspace(self.grid_range[0], self.grid_range[1], edge_samples)

        # Compute B-spline basis once
        n_knots = self.grid_size + 1 + 2 * self.spline_order
        grid = np.linspace(
            self.grid_range[0]
            - self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
            self.grid_range[1]
            + self.spline_order * (self.grid_range[1] - self.grid_range[0]) / self.grid_size,
            n_knots,
        )
        basis = np.asarray(b_spline_basis(jnp.array(x_plot), jnp.array(grid), k=self.spline_order))

        # Draw edges with activation curves
        for layer_idx, layer_key in enumerate(layer_keys):
            layer_params = params[layer_key]
            spline_weight = np.asarray(layer_params["spline_weight"])
            base_weight = np.asarray(layer_params["base_weight"])
            spline_scale = np.asarray(layer_params["spline_scale"])

            in_features, out_features, _ = spline_weight.shape

            for i in range(in_features):
                for j in range(out_features):
                    # Get node positions
                    x1, y1 = node_positions[(layer_idx, i)]
                    x2, y2 = node_positions[(layer_idx + 1, j)]

                    # Compute activation curve
                    spline_out = np.dot(basis, spline_weight[i, j, :])
                    if self.base_activation == "silu":
                        base_out = x_plot / (1 + np.exp(-x_plot)) * base_weight[i, j]
                    else:
                        base_out = x_plot * base_weight[i, j]
                    activation = spline_scale[i, j] * spline_out + base_out

                    # Normalize activation for display
                    act_range = activation.max() - activation.min()
                    if act_range > 1e-6:
                        activation_norm = (activation - activation.min()) / act_range
                    else:
                        activation_norm = np.zeros_like(activation)

                    # Draw edge as activation curve
                    # Map x_plot to edge position
                    t = np.linspace(0, 1, edge_samples)
                    edge_x = x1 + t * (x2 - x1)
                    # Perpendicular offset for curve
                    dx, dy = x2 - x1, y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 1e-6:
                        nx, ny = -dy / length, dx / length  # Normal vector
                    else:
                        nx, ny = 0, 1

                    # Scale activation curve perpendicular to edge
                    curve_scale = 0.15 * layer_spacing
                    curve_y = (activation_norm - 0.5) * curve_scale

                    edge_x_curved = edge_x + curve_y * nx
                    edge_y = y1 + t * (y2 - y1) + curve_y * ny

                    # Color based on edge strength
                    strength = np.abs(spline_scale[i, j]) + np.abs(base_weight[i, j])
                    alpha = min(0.3 + 0.7 * strength / (strength + 0.5), 1.0)

                    ax.plot(edge_x_curved, edge_y, "b-", linewidth=1.5, alpha=alpha)

        # Draw nodes
        node_radius = 0.08
        for (layer_idx, node_idx), (x, y) in node_positions.items():
            if layer_idx == 0:
                color = "lightgreen"  # Input layer
                label = f"$x_{{{node_idx}}}$" if layer_sizes[0] <= 5 else ""
            elif layer_idx == n_layers - 1:
                color = "lightcoral"  # Output layer
                if self.n_ensemble > 1:
                    out_idx = node_idx // self.n_ensemble
                    ens_idx = node_idx % self.n_ensemble
                    label = f"$y_{{{out_idx},{ens_idx}}}$" if layer_sizes[-1] <= 8 else ""
                else:
                    label = f"$y_{{{node_idx}}}$" if layer_sizes[-1] <= 5 else ""
            else:
                color = "lightblue"  # Hidden layer
                label = ""

            circle = plt.Circle((x, y), node_radius, color=color, ec="black", zorder=10)
            ax.add_patch(circle)
            if label:
                ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=11)

        # Add layer labels
        for layer_idx, n_nodes in enumerate(layer_sizes):
            x = layer_idx * layer_spacing
            y = max_nodes * node_spacing / 2 + 0.4
            if layer_idx == 0:
                ax.text(x, y, "Input", ha="center", va="bottom", fontsize=11, fontweight="bold")
            elif layer_idx == n_layers - 1:
                ax.text(x, y, "Output", ha="center", va="bottom", fontsize=11, fontweight="bold")
            else:
                ax.text(
                    x,
                    y,
                    f"Hidden {layer_idx}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

        # Set axis limits
        ax.set_xlim(-0.3, (n_layers - 1) * layer_spacing + 0.3)
        ax.set_ylim(-0.3, max_nodes * node_spacing + 0.3)

        plt.title("KAN Network Structure with Learned Activations", fontsize=14)
        plt.tight_layout()
        return fig

    def to_pyomo(self, input_bounds=None):
        """Export trained KAN to a Pyomo optimization model.

        This method creates a Pyomo model representing the trained KAN,
        enabling global optimization over the neural network using MIP solvers.

        Requirements for MIP export:
        - spline_order=1 (linear splines for piecewise linear activations)
        - base_activation='linear' (SiLU is not piecewise linear)
        - n_ensemble=1 (ensemble models cannot be exported to MIP)

        Supports multi-output models - each output becomes model.y[i].

        Args:
            input_bounds: List of (lower, upper) tuples for each input dimension.
                         If None, uses the grid_range for all inputs.

        Returns:
            pyomo.ConcreteModel: A Pyomo model with:
                - model.x[i]: Input variables
                - model.y[j]: Output variables (one per target)
                - model.obj: Objective (minimizes y[0] by default, user can modify)

        Raises:
            ValueError: If model is not MIP-compatible (wrong spline_order,
                       base_activation, or n_ensemble)
            ImportError: If pyomo is not installed

        Example:
            >>> # Train MIP-compatible KAN
            >>> kan = KAN(layers=(2, 4, 1), spline_order=1, grid_size=5,
            ...           base_activation='linear', n_ensemble=1)
            >>> kan.fit(X_train, y_train)
            >>>
            >>> # Export to Pyomo
            >>> model = kan.to_pyomo(input_bounds=[(0, 1), (0, 1)])
            >>>
            >>> # Solve for minimum
            >>> from pyomo.environ import SolverFactory
            >>> solver = SolverFactory('glpk')
            >>> result = solver.solve(model)
            >>> print(f"Optimal x: {[model.x[i].value for i in model.x]}")
            >>> print(f"Optimal y: {model.y[0].value}")
            >>>
            >>> # For multi-output, optimize a different output or combination:
            >>> model.obj.deactivate()
            >>> model.obj2 = pyo.Objective(expr=model.y[1], sense=pyo.maximize)
        """
        if self.spline_order != 1:
            raise ValueError(
                f"to_pyomo() requires spline_order=1 (linear splines) for MIP representation. "
                f"Current spline_order={self.spline_order}. "
                f"Cubic splines (order 3) are not piecewise linear and cannot be exactly "
                f"represented as mixed-integer constraints."
            )

        if self.base_activation != "linear":
            raise ValueError(
                f"to_pyomo() requires base_activation='linear' for exact MIP representation. "
                f"Current base_activation='{self.base_activation}'. "
                f"The SiLU activation is not piecewise linear. Use KAN(..., base_activation='linear') "
                f"for MIP-compatible models."
            )

        if self.n_ensemble != 1:
            raise ValueError(
                f"to_pyomo() requires n_ensemble=1 for MIP export. "
                f"Current n_ensemble={self.n_ensemble}. "
                f"Ensemble models cannot be directly exported to MIP. "
                f"Train with n_ensemble=1 for MIP-compatible models."
            )

        if not hasattr(self, "optpars"):
            raise ValueError("Model must be fitted before exporting to Pyomo.")

        try:
            import pyomo.environ as pyo
        except ImportError:
            raise ImportError("Pyomo is required for MIP export. Install with: pip install pyomo")

        # Get network dimensions
        n_inputs = self.layers[0]
        n_outputs = self.n_outputs

        # Set default input bounds
        if input_bounds is None:
            input_bounds = [self.grid_range] * n_inputs

        # Create Pyomo model
        model = pyo.ConcreteModel(name="KAN_MIP")

        # Input variables (in original space)
        model.x = pyo.Var(range(n_inputs), bounds=lambda m, i: input_bounds[i])

        # Normalized input variables
        model.x_norm = pyo.Var(range(n_inputs), within=pyo.Reals)

        # Normalization constraints
        model.norm_constraints = pyo.ConstraintList()
        for i in range(n_inputs):
            x_mean = float(self.X_mean_[i]) if self.X_mean_ is not None else 0.0
            x_std = float(self.X_std_[i]) if self.X_std_ is not None else 1.0
            # x_norm = (x - x_mean) / x_std
            model.norm_constraints.add(model.x_norm[i] == (model.x[i] - x_mean) / x_std)

        # Get breakpoints for piecewise linear formulation (in normalized space)
        n_segments = self.grid_size
        breakpoints = np.linspace(self.grid_range[0], self.grid_range[1], n_segments + 1)

        # Extract parameters
        params = self.optpars["params"]
        layer_keys = sorted([k for k in params.keys() if k.startswith("KANLayer_")])

        # Track intermediate variables for each layer
        # First layer uses normalized inputs
        layer_outputs = {-1: {i: model.x_norm[i] for i in range(n_inputs)}}

        # Big-M for constraints
        big_M = 100.0

        # Counter for unique constraint names
        constraint_counter = [0]

        def add_piecewise_constraint(model, output_var, input_var, breakpoints, values):
            """Add piecewise linear constraint using big-M formulation.

            For each segment k, we have:
                output = slope_k * input + intercept_k  when  breakpoint[k] <= input <= breakpoint[k+1]

            We use binary variables delta_k to select the active segment.
            """
            n_segs = len(breakpoints) - 1
            idx = constraint_counter[0]
            constraint_counter[0] += 1

            # Binary variables for segment selection
            delta_name = f"delta_{idx}"
            setattr(model, delta_name, pyo.Var(range(n_segs), within=pyo.Binary))
            delta = getattr(model, delta_name)

            # Lambda variables for convex combination (SOS2-like formulation)
            lam_name = f"lam_{idx}"
            setattr(
                model,
                lam_name,
                pyo.Var(range(n_segs + 1), within=pyo.NonNegativeReals, bounds=(0, 1)),
            )
            lam = getattr(model, lam_name)

            # Constraints
            constraints_name = f"pw_constraints_{idx}"
            setattr(model, constraints_name, pyo.ConstraintList())
            cons = getattr(model, constraints_name)

            # Sum of lambdas = 1
            cons.add(sum(lam[k] for k in range(n_segs + 1)) == 1)

            # Sum of deltas = 1 (exactly one segment active)
            cons.add(sum(delta[k] for k in range(n_segs)) == 1)

            # Lambda adjacency: lam[k] > 0 implies delta[k-1] or delta[k] is active
            # lam[0] <= delta[0]
            cons.add(lam[0] <= delta[0])
            # lam[n_segs] <= delta[n_segs-1]
            cons.add(lam[n_segs] <= delta[n_segs - 1])
            # lam[k] <= delta[k-1] + delta[k] for k in 1..n_segs-1
            for k in range(1, n_segs):
                cons.add(lam[k] <= delta[k - 1] + delta[k])

            # Input = sum of lambda * breakpoints
            cons.add(input_var == sum(lam[k] * breakpoints[k] for k in range(n_segs + 1)))

            # Output = sum of lambda * values
            cons.add(output_var == sum(lam[k] * values[k] for k in range(n_segs + 1)))

        for layer_idx, layer_key in enumerate(layer_keys):
            layer_params = params[layer_key]
            spline_weight = np.asarray(layer_params["spline_weight"])
            base_weight = np.asarray(layer_params["base_weight"])
            spline_scale = np.asarray(layer_params["spline_scale"])

            in_features, out_features, n_basis = spline_weight.shape

            # Create output variables for this layer
            layer_outputs[layer_idx] = {}

            for j in range(out_features):
                # Create variable for this neuron's output
                var_name = f"z_{layer_idx}_{j}"
                setattr(model, var_name, pyo.Var(within=pyo.Reals, bounds=(-big_M, big_M)))
                neuron_var = getattr(model, var_name)
                layer_outputs[layer_idx][j] = neuron_var

                # Sum contributions from all input edges
                edge_contributions = []

                for i in range(in_features):
                    input_var = layer_outputs[layer_idx - 1][i]

                    # Compute the piecewise linear function values at breakpoints
                    pw_values = []
                    for bp in breakpoints:
                        # Compute basis functions at this breakpoint
                        n_knots = self.grid_size + 1 + 2 * self.spline_order
                        grid = np.linspace(
                            self.grid_range[0]
                            - self.spline_order
                            * (self.grid_range[1] - self.grid_range[0])
                            / self.grid_size,
                            self.grid_range[1]
                            + self.spline_order
                            * (self.grid_range[1] - self.grid_range[0])
                            / self.grid_size,
                            n_knots,
                        )
                        basis = np.asarray(
                            b_spline_basis(jnp.array([bp]), jnp.array(grid), k=self.spline_order)
                        )

                        # Spline contribution
                        spline_val = float(np.dot(basis[0], spline_weight[i, j, :]))

                        # Base contribution (for linear splines, use linear base)
                        base_val = float(bp * base_weight[i, j])

                        # Combined value
                        combined = float(spline_scale[i, j]) * spline_val + base_val
                        pw_values.append(combined)

                    # Create edge output variable
                    edge_var_name = f"edge_{layer_idx}_{i}_{j}"
                    edge_lb = min(pw_values) - 1.0
                    edge_ub = max(pw_values) + 1.0
                    setattr(
                        model, edge_var_name, pyo.Var(within=pyo.Reals, bounds=(edge_lb, edge_ub))
                    )
                    edge_var = getattr(model, edge_var_name)
                    edge_contributions.append(edge_var)

                    # Add piecewise linear constraint
                    add_piecewise_constraint(model, edge_var, input_var, breakpoints, pw_values)

                # Sum all edge contributions for this neuron
                sum_constraint_name = f"sum_{layer_idx}_{j}"
                setattr(
                    model,
                    sum_constraint_name,
                    pyo.Constraint(expr=neuron_var == sum(edge_contributions)),
                )

        # Output variables (denormalized) - one per output
        model.y = pyo.Var(range(n_outputs), within=pyo.Reals)

        # Denormalization constraints for each output
        model.denorm_constraints = pyo.ConstraintList()
        for j in range(n_outputs):
            final_output = layer_outputs[len(layer_keys) - 1][j]

            # Denormalization: y = y_norm * y_std + y_mean
            y_mean = float(self.y_mean_[j]) if self.y_mean_ is not None else 0.0
            y_std = float(self.y_std_[j]) if self.y_std_ is not None else 1.0
            model.denorm_constraints.add(model.y[j] == final_output * y_std + y_mean)

        # Default objective: minimize first output (user can modify)
        model.obj = pyo.Objective(expr=model.y[0], sense=pyo.minimize)

        return model
