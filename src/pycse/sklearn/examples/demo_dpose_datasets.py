"""Demonstration of DPOSE on Various Datasets

This script showcases DPOSE performance on different types of:
1. Heteroscedastic noise patterns
2. Nonlinear functions
3. Challenging scenarios

Generates comprehensive visualizations showing predictions with uncertainty bands.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pycse.sklearn.dpose import DPOSE


def generate_datasets(n_samples=200, seed=42):
    """Generate diverse datasets for testing DPOSE."""

    key = jax.random.PRNGKey(seed)
    datasets = {}

    # Dataset 1: Linear with increasing noise (classic heteroscedastic)
    x1 = np.linspace(0, 1, n_samples)[:, None]
    noise1 = 0.01 + 0.15 * x1.ravel()
    y1 = 2 * x1.ravel() + noise1 * jax.random.normal(key, (n_samples,))
    datasets["Linear + Increasing Noise"] = (x1, y1, noise1)

    # Dataset 2: Quadratic with noise proportional to function value
    key, subkey = jax.random.split(key)
    x2 = np.linspace(-1, 1, n_samples)[:, None]
    f2 = 3 * x2.ravel() ** 2 + 1
    noise2 = 0.1 * f2  # Noise scales with function value
    y2 = f2 + noise2 * jax.random.normal(subkey, (n_samples,))
    datasets["Quadratic + Proportional Noise"] = (x2, y2, noise2)

    # Dataset 3: Sine wave with position-dependent noise
    key, subkey = jax.random.split(key)
    x3 = np.linspace(0, 2 * np.pi, n_samples)[:, None]
    f3 = np.sin(x3.ravel())
    noise3 = 0.05 + 0.3 * np.abs(np.sin(2 * x3.ravel()))  # Noise varies with 2x frequency
    y3 = f3 + noise3 * jax.random.normal(subkey, (n_samples,))
    datasets["Sine Wave + Periodic Noise"] = (x3, y3, noise3)

    # Dataset 4: Exponential with heteroscedastic noise
    key, subkey = jax.random.split(key)
    x4 = np.linspace(0, 1, n_samples)[:, None]
    f4 = np.exp(2 * x4.ravel())
    noise4 = 0.1 * f4  # Noise grows with function value
    y4 = f4 + noise4 * jax.random.normal(subkey, (n_samples,))
    datasets["Exponential + Growing Noise"] = (x4, y4, noise4)

    # Dataset 5: Cubic with noise dip in middle
    key, subkey = jax.random.split(key)
    x5 = np.linspace(-1, 1, n_samples)[:, None]
    f5 = x5.ravel() ** 3 + 2 * x5.ravel()
    noise5 = 0.1 + 0.2 * (1 - 4 * (x5.ravel()) ** 2)  # Noise minimum at center
    noise5 = np.maximum(noise5, 0.05)
    y5 = f5 + noise5 * jax.random.normal(subkey, (n_samples,))
    datasets["Cubic + Variable Noise"] = (x5, y5, noise5)

    # Dataset 6: Step function with different noise levels
    key, subkey = jax.random.split(key)
    x6 = np.linspace(0, 1, n_samples)[:, None]
    f6 = np.where(x6.ravel() < 0.5, 0.5, 1.5)
    noise6 = np.where(x6.ravel() < 0.5, 0.05, 0.15)  # Different noise in each regime
    y6 = f6 + noise6 * jax.random.normal(subkey, (n_samples,))
    datasets["Step Function + Level-Dependent Noise"] = (x6, y6, noise6)

    # Dataset 7: Square root with decreasing noise
    key, subkey = jax.random.split(key)
    x7 = np.linspace(0.1, 1, n_samples)[:, None]
    f7 = np.sqrt(x7.ravel())
    noise7 = 0.15 * (1 - x7.ravel())  # Noise decreases with x
    noise7 = np.maximum(noise7, 0.02)
    y7 = f7 + noise7 * jax.random.normal(subkey, (n_samples,))
    datasets["Square Root + Decreasing Noise"] = (x7, y7, noise7)

    # Dataset 8: Logarithm with complex noise pattern
    key, subkey = jax.random.split(key)
    x8 = np.linspace(0.1, 2, n_samples)[:, None]
    f8 = np.log(x8.ravel())
    noise8 = 0.05 + 0.1 * np.sin(4 * np.pi * x8.ravel()) ** 2  # Oscillating noise
    y8 = f8 + noise8 * jax.random.normal(subkey, (n_samples,))
    datasets["Logarithm + Oscillating Noise"] = (x8, y8, noise8)

    # Dataset 9: Rational function with extreme heteroscedasticity
    key, subkey = jax.random.split(key)
    x9 = np.linspace(-2, 2, n_samples)[:, None]
    x9 = x9[np.abs(x9.ravel()) > 0.2]  # Remove near-singularity
    f9 = 1 / (1 + x9.ravel() ** 2)
    noise9 = 0.02 + 0.2 * f9  # Noise proportional to function value
    y9 = f9 + noise9 * jax.random.normal(subkey, (len(x9),))
    datasets["Rational Function + Extreme Heteroscedasticity"] = (x9, y9, noise9)

    # Dataset 10: Polynomial with multi-modal noise
    key, subkey = jax.random.split(key)
    x10 = np.linspace(-1, 1, n_samples)[:, None]
    f10 = x10.ravel() ** 4 - x10.ravel() ** 2
    noise10 = 0.03 + 0.1 * np.abs(x10.ravel())  # Noise increases away from center
    y10 = f10 + noise10 * jax.random.normal(subkey, (n_samples,))
    datasets["Quartic + Multi-modal Noise"] = (x10, y10, noise10)

    return datasets


def fit_and_visualize(datasets, loss_type="crps", activation_name="relu"):
    """Fit DPOSE to all datasets and create comprehensive visualization."""

    from flax import linen as nn

    # Map activation names to functions
    activation_map = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "softplus": nn.softplus,
        "elu": nn.elu,
        "gelu": nn.gelu,
    }
    activation = activation_map.get(activation_name, nn.relu)

    n_datasets = len(datasets)
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()

    print(f"\n{'='*80}")
    print(f"DPOSE DEMONSTRATION: {loss_type.upper()} Loss, {activation_name.upper()} Activation")
    print(f"{'='*80}\n")

    for idx, (name, (x, y, true_noise)) in enumerate(datasets.items()):
        print(f"\nDataset {idx+1}/{n_datasets}: {name}")
        print(f"{'─'*80}")

        ax = axes[idx]

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Fit DPOSE with Pipeline and StandardScaler
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "dpose",
                    DPOSE(
                        layers=(1, 20, 32),
                        loss_type=loss_type,
                        activation=activation,
                        maxiter=500,
                        seed=42,
                    ),
                ),
            ]
        )

        model.fit(x_train, y_train)

        # Predictions
        x_plot = np.sort(x, axis=0)
        y_pred, y_std = model.predict(x_plot, return_std=True)

        # Compute metrics
        y_test_pred = model.predict(x_test)
        mae = np.abs(y_test - y_test_pred).mean()
        rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))

        print(f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        print(f"  Uncertainty range: [{y_std.min():.4f}, {y_std.max():.4f}]")

        # Check heteroscedasticity
        uncertainty_range = y_std.max() - y_std.min()
        if uncertainty_range / y_std.mean() > 0.5:
            print(f"  ✓ Captured heteroscedastic uncertainty (range={uncertainty_range:.4f})")
        else:
            print(f"  ⚠ Uncertainties relatively uniform (range={uncertainty_range:.4f})")

        # Plot
        # Sort everything by x for smooth plotting
        sort_idx = np.argsort(x_plot.ravel())
        x_sorted = x_plot.ravel()[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]

        # Uncertainty band (95% confidence interval)
        ax.fill_between(
            x_sorted,
            y_pred_sorted - 2 * y_std_sorted,
            y_pred_sorted + 2 * y_std_sorted,
            alpha=0.3,
            color="red",
            label="±2σ (95% CI)",
            zorder=1,
        )

        # Mean prediction
        ax.plot(x_sorted, y_pred_sorted, "r-", linewidth=2.5, label="DPOSE prediction", zorder=3)

        # Training data
        ax.scatter(x_train, y_train, alpha=0.5, s=20, color="blue", label="Training data", zorder=4)

        # Test data
        ax.scatter(
            x_test, y_test, alpha=0.7, s=40, color="green", marker="s", label="Test data", zorder=5
        )

        # Formatting
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.set_title(
            f"{name}\nMAE={mae:.4f}, σ∈[{y_std.min():.3f}, {y_std.max():.3f}]", fontsize=10, pad=10
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = f"dpose_demo_{loss_type}_{activation_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\n{'='*80}")
    print(f"✓ Saved visualization to '{filename}'")
    print(f"{'='*80}\n")

    return fig


def compare_loss_functions(dataset_name="Linear + Increasing Noise", n_samples=200):
    """Compare CRPS vs NLL on a single dataset."""

    datasets = generate_datasets(n_samples=n_samples)
    x, y, true_noise = datasets[dataset_name]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    print(f"\n{'='*80}")
    print(f"COMPARING LOSS FUNCTIONS: {dataset_name}")
    print(f"{'='*80}\n")

    for idx, loss_type in enumerate(["crps", "nll"]):
        ax = axes[idx]

        print(f"{loss_type.upper()} Loss:")

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("dpose", DPOSE(layers=(1, 20, 32), loss_type=loss_type, maxiter=500, seed=42)),
            ]
        )

        model.fit(x_train, y_train)

        # Predictions
        x_plot = np.sort(x, axis=0)
        y_pred, y_std = model.predict(x_plot, return_std=True)

        # Metrics
        y_test_pred = model.predict(x_test)
        mae = np.abs(y_test - y_test_pred).mean()

        print(f"  MAE: {mae:.6f}")
        print(f"  Uncertainty: [{y_std.min():.4f}, {y_std.max():.4f}]\n")

        # Sort for plotting
        sort_idx = np.argsort(x_plot.ravel())
        x_sorted = x_plot.ravel()[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]

        # Plot
        ax.fill_between(
            x_sorted,
            y_pred_sorted - 2 * y_std_sorted,
            y_pred_sorted + 2 * y_std_sorted,
            alpha=0.3,
            color="red",
            label="±2σ",
        )
        ax.plot(x_sorted, y_pred_sorted, "r-", linewidth=2.5, label="Prediction")
        ax.scatter(x_train, y_train, alpha=0.5, s=20, color="blue", label="Training")
        ax.scatter(x_test, y_test, alpha=0.7, s=40, color="green", marker="s", label="Test")

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(f"{loss_type.upper()} Loss\nMAE={mae:.4f}", fontsize=12, pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dpose_loss_comparison.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved comparison to 'dpose_loss_comparison.png'\n")

    return fig


def demonstrate_uncertainty_propagation(n_samples=100):
    """Demonstrate uncertainty propagation through nonlinear transformation."""

    print(f"\n{'='*80}")
    print("UNCERTAINTY PROPAGATION DEMO")
    print(f"{'='*80}\n")

    # Generate data
    key = jax.random.PRNGKey(42)
    x = np.linspace(0, 1, n_samples)[:, None]
    noise = 0.05 + 0.1 * x.ravel()
    y = x.ravel() + noise * jax.random.normal(key, (n_samples,))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit model with Pipeline and StandardScaler
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("dpose", DPOSE(layers=(1, 15, 32), loss_type="crps", maxiter=500, seed=42)),
        ]
    )
    model.fit(x_train, y_train)

    # Get ensemble predictions
    x_plot = np.sort(x, axis=0)
    # Need to access DPOSE through pipeline for custom predict_ensemble method
    x_plot_scaled = model.named_steps["scaler"].transform(x_plot)
    ensemble_preds = model.named_steps["dpose"].predict_ensemble(x_plot_scaled)

    # Apply nonlinear transformation: z = exp(y)
    z_ensemble = np.exp(ensemble_preds)
    z_mean = z_ensemble.mean(axis=1)
    z_std = z_ensemble.std(axis=1)

    # Compare with naive approach (linear approximation)
    y_mean, y_std = model.predict(x_plot, return_std=True)
    z_mean_naive = np.exp(y_mean)
    z_std_naive = z_mean_naive * y_std  # First-order approximation

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original space
    ax = axes[0]
    sort_idx = np.argsort(x_plot.ravel())
    x_sorted = x_plot.ravel()[sort_idx]
    y_mean_sorted = y_mean[sort_idx]
    y_std_sorted = y_std[sort_idx]

    ax.fill_between(
        x_sorted,
        y_mean_sorted - 2 * y_std_sorted,
        y_mean_sorted + 2 * y_std_sorted,
        alpha=0.3,
        color="blue",
        label="±2σ",
    )
    ax.plot(x_sorted, y_mean_sorted, "b-", linewidth=2.5, label="Mean")
    ax.scatter(x, y, alpha=0.5, s=20, color="black", label="Data")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Original Space: y = f(x)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Transformed space
    ax = axes[1]
    z_mean_sorted = z_mean[sort_idx]
    z_std_sorted = z_std[sort_idx]
    z_mean_naive_sorted = z_mean_naive[sort_idx]
    z_std_naive_sorted = z_std_naive[sort_idx]

    # Ensemble propagation
    ax.fill_between(
        x_sorted,
        z_mean_sorted - 2 * z_std_sorted,
        z_mean_sorted + 2 * z_std_sorted,
        alpha=0.3,
        color="red",
        label="±2σ (ensemble)",
    )
    ax.plot(x_sorted, z_mean_sorted, "r-", linewidth=2.5, label="Ensemble mean")

    # Naive propagation
    ax.plot(x_sorted, z_mean_naive_sorted, "g--", linewidth=2, label="Naive mean", alpha=0.7)
    ax.fill_between(
        x_sorted,
        z_mean_naive_sorted - 2 * z_std_naive_sorted,
        z_mean_naive_sorted + 2 * z_std_naive_sorted,
        alpha=0.2,
        color="green",
        label="±2σ (naive)",
        linestyle="--",
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("z = exp(y)", fontsize=12)
    ax.set_title("Transformed Space: z = exp(y)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dpose_uncertainty_propagation.png", dpi=150, bbox_inches="tight")

    print("Transformed uncertainties:")
    print(f"  Ensemble propagation: σ_z ∈ [{z_std.min():.4f}, {z_std.max():.4f}]")
    print(f"  Naive (linear):       σ_z ∈ [{z_std_naive.min():.4f}, {z_std_naive.max():.4f}]")
    print(f"  Difference: {np.abs(z_std - z_std_naive).mean():.4f} (mean absolute)")
    print(f"\n✓ Saved to 'dpose_uncertainty_propagation.png'\n")

    return fig


if __name__ == "__main__":
    # Generate all datasets
    print("\nGenerating diverse heteroscedastic datasets...")
    datasets = generate_datasets(n_samples=200)
    print(f"✓ Created {len(datasets)} datasets\n")

    # Main demonstration with CRPS
    print("\n" + "=" * 80)
    print("MAIN DEMONSTRATION")
    print("=" * 80)
    fit_and_visualize(datasets, loss_type="crps", activation_name="relu")

    # Compare loss functions
    compare_loss_functions()

    # Uncertainty propagation demo
    demonstrate_uncertainty_propagation()

    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. dpose_demo_crps_relu.png       - Main comprehensive demo")
    print("  2. dpose_loss_comparison.png      - CRPS vs NLL comparison")
    print("  3. dpose_uncertainty_propagation.png - Nonlinear transformation")
    print("\n" + "=" * 80 + "\n")
