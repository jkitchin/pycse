"""Compare NNBR vs DPOSE on heteroscedastic regression.

This script provides a fair comparison between:
- NNBR: Neural Network + Bayesian Ridge (sklearn-based)
- DPOSE: Direct Propagation of Shallow Ensembles (JAX-based)

Both models now have equivalent diagnostic capabilities (plots, metrics, reports),
so any performance differences reflect the underlying methodology, not tooling.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

# Import both models
import sys

sys.path.insert(0, "..")
from nnbr import NeuralNetworkBLR
from dpose import DPOSE


def generate_heteroscedastic_data(n_samples=200, noise_range=(0.01, 0.08), seed=42):
    """Generate data with input-dependent noise (heteroscedastic).

    Args:
        n_samples: Number of samples to generate
        noise_range: (min_noise, max_noise) tuple
        seed: Random seed

    Returns:
        X, y, true_noise arrays
    """
    key = jax.random.PRNGKey(seed)
    X = np.linspace(0, 1, n_samples)[:, None]

    # True function: y = x^(1/3)
    y_true = X.ravel() ** (1 / 3)

    # Heteroscedastic noise: increases with X
    noise_std = noise_range[0] + noise_range[1] * X.ravel()
    noise = noise_std * jax.random.normal(key, (n_samples,))
    y = y_true + noise

    return X, y, noise_std


def compare_models():
    """Run comprehensive comparison between NNBR and DPOSE."""

    print("=" * 70)
    print("NNBR vs DPOSE Comparison")
    print("=" * 70)
    print("\nGenerating heteroscedastic data (n=200)...")

    # Generate data
    X, y, true_noise = generate_heteroscedastic_data(n_samples=200, seed=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  True noise range: {true_noise.min():.4f} to {true_noise.max():.4f}")

    # ========================================================================
    # NNBR (sklearn-based)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training NNBR (Neural Network + Bayesian Ridge)")
    print("=" * 70)

    # Setup comparable architecture to DPOSE
    nn = MLPRegressor(
        hidden_layer_sizes=(20,),  # Single hidden layer with 20 units
        activation="relu",
        solver="lbfgs",
        max_iter=1000,
        random_state=19,
    )

    br = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)

    nnbr = NeuralNetworkBLR(nn, br)
    nnbr.fit(X_train, y_train, val_X=X_val, val_y=y_val)

    print("\n--- NNBR Report ---")
    nnbr.report()

    print("\n--- NNBR Validation Metrics ---")
    nnbr.print_metrics(X_val, y_val)

    # ========================================================================
    # DPOSE (JAX-based)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training DPOSE (Direct Propagation of Shallow Ensembles)")
    print("=" * 70)

    # Use comparable architecture: (1, 20, 32)
    # - Input: 1
    # - Hidden: 20 (same as NNBR)
    # - Ensemble: 32 outputs
    dpose = DPOSE(
        layers=(1, 20, 32),
        loss_type="crps",  # Use CRPS for robustness
        optimizer="bfgs",  # Use BFGS for fair comparison with sklearn's lbfgs
        seed=19,
    )

    dpose.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=1000)

    print("\n--- DPOSE Report ---")
    dpose.report()

    print("\n--- DPOSE Validation Metrics ---")
    dpose.print_metrics(X_val, y_val)

    # ========================================================================
    # Side-by-Side Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE METRICS COMPARISON")
    print("=" * 70)

    nnbr_metrics = nnbr.uncertainty_metrics(X_val, y_val)
    dpose_metrics = dpose.uncertainty_metrics(X_val, y_val)

    print(f"\n{'Metric':<25} {'NNBR':<15} {'DPOSE':<15} {'Winner':<10}")
    print("-" * 70)

    # RMSE (lower is better)
    print(f"{'RMSE':<25} {nnbr_metrics['rmse']:<15.6f} {dpose_metrics['rmse']:<15.6f} ", end="")
    if nnbr_metrics["rmse"] < dpose_metrics["rmse"]:
        print("NNBR ✓")
    else:
        print("DPOSE ✓")

    # MAE (lower is better)
    print(f"{'MAE':<25} {nnbr_metrics['mae']:<15.6f} {dpose_metrics['mae']:<15.6f} ", end="")
    if nnbr_metrics["mae"] < dpose_metrics["mae"]:
        print("NNBR ✓")
    else:
        print("DPOSE ✓")

    # NLL (lower is better)
    if not np.isnan(nnbr_metrics["nll"]) and not np.isnan(dpose_metrics["nll"]):
        print(f"{'NLL':<25} {nnbr_metrics['nll']:<15.6f} {dpose_metrics['nll']:<15.6f} ", end="")
        if nnbr_metrics["nll"] < dpose_metrics["nll"]:
            print("NNBR ✓")
        else:
            print("DPOSE ✓")
    else:
        print(f"{'NLL':<25} {'N/A':<15} {'N/A':<15} {'N/A':<10}")

    # Miscalibration (lower is better)
    if not np.isnan(nnbr_metrics["miscalibration_area"]) and not np.isnan(
        dpose_metrics["miscalibration_area"]
    ):
        print(
            f"{'Miscalibration Area':<25} {nnbr_metrics['miscalibration_area']:<15.6f} {dpose_metrics['miscalibration_area']:<15.6f} ",
            end="",
        )
        if nnbr_metrics["miscalibration_area"] < dpose_metrics["miscalibration_area"]:
            print("NNBR ✓")
        else:
            print("DPOSE ✓")

    # Z-score std (closer to 1.0 is better)
    if not np.isnan(nnbr_metrics["z_score_std"]) and not np.isnan(dpose_metrics["z_score_std"]):
        print(
            f"{'Z-score std (ideal=1)':<25} {nnbr_metrics['z_score_std']:<15.6f} {dpose_metrics['z_score_std']:<15.6f} ",
            end="",
        )
        nnbr_diff = abs(nnbr_metrics["z_score_std"] - 1.0)
        dpose_diff = abs(dpose_metrics["z_score_std"] - 1.0)
        if nnbr_diff < dpose_diff:
            print("NNBR ✓")
        else:
            print("DPOSE ✓")

    print("\n" + "=" * 70)

    # ========================================================================
    # Visualization
    # ========================================================================
    print("\nGenerating comparison plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # NNBR plot
    nnbr.plot(X, y, ax=axes[0])
    axes[0].set_title(
        f"NNBR (sklearn)\nRMSE={nnbr_metrics['rmse']:.4f}, MAE={nnbr_metrics['mae']:.4f}"
    )

    # DPOSE plot
    dpose.plot(X, y, ax=axes[1], distribution=False)
    axes[1].set_title(
        f"DPOSE (JAX)\nRMSE={dpose_metrics['rmse']:.4f}, MAE={dpose_metrics['mae']:.4f}"
    )

    plt.tight_layout()
    plt.savefig("nnbr_vs_dpose_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved: nnbr_vs_dpose_comparison.png")

    # Additional plot: Uncertainty vs X
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get predictions
    nnbr_pred, nnbr_std = nnbr.predict(X, return_std=True)
    dpose_pred, dpose_std = dpose.predict(X, return_std=True)

    X_sorted = X.ravel()
    sort_idx = np.argsort(X_sorted)

    ax.plot(X_sorted[sort_idx], nnbr_std[sort_idx], "b-", label="NNBR", linewidth=2)
    ax.plot(X_sorted[sort_idx], dpose_std[sort_idx], "r-", label="DPOSE", linewidth=2)
    ax.plot(
        X_sorted[sort_idx], true_noise[sort_idx], "k--", label="True noise", linewidth=2, alpha=0.7
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Predicted Uncertainty (σ)")
    ax.set_title("Uncertainty Estimates: NNBR vs DPOSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("uncertainty_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved: uncertainty_comparison.png")

    plt.show()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Differences:")
    print("  NNBR:")
    print("    - Backend: sklearn (CPU-optimized)")
    print("    - Method: Single NN + Bayesian Ridge last layer")
    print("    - Uncertainty: Analytical (from Bayesian Ridge)")
    print("    - Calibration: Post-hoc α scaling")
    print(f"    - Calibration factor: {nnbr.calibration_factor:.4f}")
    print("\n  DPOSE:")
    print("    - Backend: JAX/Flax (GPU-ready)")
    print("    - Method: Shallow ensemble (32 last-layer outputs)")
    print("    - Uncertainty: Ensemble variance")
    print("    - Calibration: Post-hoc α scaling")
    print(f"    - Calibration factor: {dpose.calibration_factor:.4f}")
    print("\nBoth models now have equivalent diagnostic tools.")
    print("Performance differences reflect methodology, not tooling.")
    print("=" * 70)


if __name__ == "__main__":
    compare_models()
