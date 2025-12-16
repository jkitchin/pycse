"""Test 5D padding approach."""

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

print("=" * 70)
print("Testing 5D Padding Approach")
print("=" * 70)

# Test heteroskedastic
print("\nHeteroskedastic Regression with 5D Padding")
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

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: Predictions
ax = axes[0]
ax.scatter(X_het, y_het, alpha=0.3, s=10, label="Data", c="gray")
ax.plot(X_het, y_pred_het, "r-", label="Mean prediction", linewidth=2)
ax.fill_between(
    X_het.ravel(),
    (y_pred_het - 2 * y_std_het).ravel(),
    (y_pred_het + 2 * y_std_het).ravel(),
    alpha=0.3,
    color="red",
    label="95% confidence",
)
ax.plot(X_het, y_true_het, "k--", label="True function", linewidth=1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Heteroskedastic Regression (MSE={mse:.4f})")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Uncertainty comparison
ax = axes[1]
ax.plot(X_het, noise_std * 2, "k--", label="True noise (2σ)", linewidth=2)
ax.plot(X_het, y_std_het * 2, "r-", label="Learned uncertainty (2σ)", linewidth=2)
ax.set_xlabel("X")
ax.set_ylabel("Uncertainty (2σ)")
ax.set_title("Uncertainty Learning")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "/Users/jkitchin/Dropbox/python/pycse/heteroskedastic_5d.png", dpi=150, bbox_inches="tight"
)
print("\nPlot saved to heteroskedastic_5d.png")
plt.close()

# Diagnosis
print("\n" + "=" * 70)
print("RESULTS with 5D Padding")
print("=" * 70)

if mse < 0.5:
    print("✓ Mean prediction is GOOD")
else:
    print(f"❌ Mean prediction is still poor (MSE={mse:.4f})")

if 0.7 <= avg_ratio <= 1.3:
    print(f"✓ Uncertainty magnitude is reasonable (avg ratio={avg_ratio:.2f})")
else:
    print(f"❌ Uncertainty is off (avg ratio={avg_ratio:.2f})")

# Check if heteroskedasticity is learned
std_at_center = y_std_het[len(X_het) // 2, 0]
std_at_edges = (y_std_het[0, 0] + y_std_het[-1, 0]) / 2

ratio_het = std_at_edges / std_at_center
if ratio_het > 1.5:
    print(f"✓ Heteroskedasticity is learned (edge/center ratio={ratio_het:.2f})")
else:
    print(f"❌ Heteroskedasticity not learned (edge/center ratio={ratio_het:.2f})")

print("=" * 70)
