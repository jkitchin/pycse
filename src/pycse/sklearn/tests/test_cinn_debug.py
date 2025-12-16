"""Debug script to understand CINN 1D padding behavior."""

import jax
import jax.numpy as np

# Enable 64-bit
jax.config.update("jax_enable_x64", True)


def analyze_split_pattern():
    """Analyze how splits work with 1D->3D padding."""

    n_features_out = 3  # Internal dimension after padding 1D
    n_layers = 8

    print("=" * 70)
    print("CINN 1D Padding Analysis")
    print("=" * 70)
    print("\n1D output is padded to 3D: [padding_0, Y, padding_2]")
    print("Y is in position 1 (middle)")
    print(f"\nNumber of layers: {n_layers}")
    print("\nSplit pattern per layer:")
    print("-" * 70)

    for i in range(n_layers):
        # This is the split logic from _ConditionalFlowModel
        if i % 2 == 0:
            split_idx = n_features_out // 2
        else:
            split_idx = n_features_out - n_features_out // 2

        unchanged_dims = f"[0:{split_idx}]"
        transform_dims = f"[{split_idx}:3]"

        # Determine which actual dimensions
        transform_list = list(range(split_idx, 3))

        y_status = "TRANSFORMED" if 1 in transform_list else "UNCHANGED"

        print(
            f"Layer {i} (split={split_idx}): unchanged={unchanged_dims}, "
            f"transform={transform_dims} -> Y is {y_status}"
        )

    # Count how many times Y gets transformed
    y_transform_count = 0
    for i in range(n_layers):
        if i % 2 == 0:
            split_idx = n_features_out // 2  # = 1
        else:
            split_idx = n_features_out - n_features_out // 2  # = 2

        if 1 >= split_idx:  # Y is in position 1
            y_transform_count += 1

    print("\n" + "=" * 70)
    print(f"ISSUE: Y gets transformed in only {y_transform_count}/{n_layers} layers!")
    print("On odd layers, Y is UNCHANGED (only used to transform padding)")
    print("=" * 70)
    print("\nThis limits the model's expressiveness significantly!")


def test_simple_fit():
    """Test actual fitting behavior."""
    from pycse.sklearn.cinn import ConditionalInvertibleNN

    print("\n\n" + "=" * 70)
    print("Testing Simple Linear Fit")
    print("=" * 70)

    # Simple linear data
    key = jax.random.PRNGKey(42)
    X = np.linspace(-3, 3, 100)[:, None]
    y_true = 2 * X + 1
    y = y_true + 0.1 * jax.random.normal(key, X.shape)

    # Train
    cinn = ConditionalInvertibleNN(
        n_features_in=1, n_features_out=1, n_layers=8, hidden_dims=[128, 128], seed=42
    )

    print("\nData: y = 2*x + 1 with noise=0.1")
    print("Training...")
    cinn.fit(X, y, maxiter=1000)

    # Check predictions
    y_pred = cinn.predict(X)
    mse = np.mean((y_pred - y_true) ** 2)

    print("\nResults:")
    print(f"  Final NLL: {cinn.final_nll_:.4f}")
    print(f"  MSE vs true function: {mse:.4f}")
    print("  Expected MSE (noise only): ~0.01")

    if mse > 0.05:
        print("  ⚠️  WARNING: MSE is much higher than expected!")

    # Check if model learned the right slope
    X_test = np.array([[-2.0], [0.0], [2.0]])
    y_test_true = 2 * X_test + 1
    y_test_pred = cinn.predict(X_test)

    print("\nTest predictions:")
    for i in range(len(X_test)):
        print(
            f"  X={X_test[i, 0]:5.1f}: true={y_test_true[i, 0]:6.2f}, "
            f"pred={y_test_pred[i, 0]:6.2f}, error={abs(y_test_pred[i, 0] - y_test_true[i, 0]):.3f}"
        )


if __name__ == "__main__":
    analyze_split_pattern()
    test_simple_fit()
