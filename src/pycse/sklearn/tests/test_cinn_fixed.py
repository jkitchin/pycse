"""Test the fixed CINN implementation."""

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN

jax.config.update("jax_enable_x64", True)

print("=" * 70)
print("Testing FIXED Implementation")
print("=" * 70)

# Test 1: Simple linear
print("\n1. Simple Linear Regression")
print("-" * 70)
key = jax.random.PRNGKey(42)
X = np.linspace(-3, 3, 100)[:, None]
y_true = 2 * X + 1
y = y_true + 0.1 * jax.random.normal(key, X.shape)

cinn = ConditionalInvertibleNN(
    n_features_in=1, n_features_out=1, n_layers=8, hidden_dims=[128, 128], seed=42
)

print("Training...")
cinn.fit(X, y, maxiter=1000)

y_pred = cinn.predict(X)
mse = np.mean((y_pred - y_true) ** 2)

print(f"Final NLL: {cinn.final_nll_:.4f}")
print(f"MSE vs true: {mse:.4f} (expected ~0.01)")

if mse < 0.05:
    print("✓ Linear fit is GOOD!")
else:
    print("❌ Linear fit still poor")

# Test 2: Heteroskedastic
print("\n2. Heteroskedastic Regression")
print("-" * 70)
key = jax.random.PRNGKey(99)
X_het = np.linspace(-3, 3, 250)[:, None]
y_true_het = X_het**2

noise_std = 0.1 + 0.3 * np.abs(X_het)
noise = noise_std * jax.random.normal(key, X_het.shape)
y_het = y_true_het + noise

cinn_het = ConditionalInvertibleNN(
    n_features_in=1, n_features_out=1, n_layers=10, hidden_dims=[128, 128, 128], seed=42
)

print("Training...")
cinn_het.fit(X_het, y_het, maxiter=2500)

y_pred_het, y_std_het = cinn_het.predict(X_het, return_std=True, n_samples=1000)

mse = np.mean((y_pred_het - y_true_het) ** 2)
print(f"\nFinal NLL: {cinn_het.final_nll_:.4f}")
print(f"MSE vs true: {mse:.4f}")

# Check uncertainty at different points
test_indices = [25, 75, 125, 175, 225]
print("\nUncertainty Comparison:")
print("X      True σ   Learned σ   Ratio")
print("-" * 40)

ratios = []
for idx in test_indices:
    x_val = X_het[idx, 0]
    true_sigma = noise_std[idx, 0]
    learned_sigma = y_std_het[idx, 0]
    ratio = learned_sigma / true_sigma if true_sigma > 0 else 0
    ratios.append(ratio)
    print(f"{x_val:5.2f}  {true_sigma:7.3f}  {learned_sigma:10.3f}  {ratio:6.2f}")

avg_ratio = np.mean(np.array(ratios))

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

if mse < 0.5:
    print("✓ Mean prediction is GOOD")
else:
    print("❌ Mean prediction is still poor")

if 0.7 <= avg_ratio <= 1.3:
    print("✓ Uncertainty magnitude is reasonable")
else:
    print(f"❌ Uncertainty is off (avg ratio: {avg_ratio:.2f})")

# Check if heteroskedasticity is learned
std_at_center = y_std_het[len(X_het) // 2, 0]
std_at_edges = (y_std_het[0, 0] + y_std_het[-1, 0]) / 2

if std_at_edges > 1.5 * std_at_center:
    print("✓ Heteroskedasticity is learned (uncertainty varies with X)")
else:
    print(
        f"❌ Heteroskedasticity not learned (center={std_at_center:.3f}, edges={std_at_edges:.3f})"
    )

print("=" * 70)
