"""Test heteroskedastic fitting."""

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# Generate heteroskedastic data
key = jax.random.PRNGKey(99)
X_het = np.linspace(-3, 3, 250)[:, None]
y_true_het = X_het**2

# Noise increases with |X|
noise_std = 0.1 + 0.3 * np.abs(X_het)
noise = noise_std * jax.random.normal(key, X_het.shape)
y_het = y_true_het + noise

print("Heteroskedastic Data:")
print(f"  Noise at X=0: ~{noise_std[len(X_het) // 2, 0]:.2f}")
print(f"  Noise at X=±3: ~{noise_std[-1, 0]:.2f}")

# Train conditional flow
cinn_het = ConditionalInvertibleNN(
    n_features_in=1, n_features_out=1, n_layers=10, hidden_dims=[128, 128, 128], seed=42
)

print("\nTraining...")
cinn_het.fit(X_het, y_het, maxiter=2500)

print(f"\nFinal NLL: {cinn_het.final_nll_:.4f}")
print(f"Converged: {'Yes' if cinn_het.state_.iter_num < cinn_het.maxiter else 'No'}")

# Predict with uncertainty
y_pred_het, y_std_het = cinn_het.predict(X_het, return_std=True, n_samples=1000)

# Check mean prediction quality
mse = np.mean((y_pred_het - y_true_het) ** 2)
print(f"\nMSE vs true function: {mse:.4f}")

# Check if uncertainty is learned correctly
# Compare learned std to true std at several points
test_indices = [25, 75, 125, 175, 225]  # Different X values
print("\nUncertainty Comparison:")
print("X      True σ   Learned σ   Ratio")
print("-" * 40)
for idx in test_indices:
    x_val = X_het[idx, 0]
    true_sigma = noise_std[idx, 0]
    learned_sigma = y_std_het[idx, 0]
    ratio = learned_sigma / true_sigma if true_sigma > 0 else 0
    print(f"{x_val:5.2f}  {true_sigma:7.3f}  {learned_sigma:10.3f}  {ratio:6.2f}")

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
    "/Users/jkitchin/Dropbox/python/pycse/heteroskedastic_test.png", dpi=150, bbox_inches="tight"
)
print("\nPlot saved to heteroskedastic_test.png")
plt.close()

# Diagnosis
print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

if mse > 0.5:
    print("❌ Mean prediction is POOR (high MSE)")
else:
    print("✓ Mean prediction is reasonable")

# Check if uncertainty increases with |X|
std_at_center = y_std_het[len(X_het) // 2, 0]
std_at_edge = np.mean([y_std_het[0, 0], y_std_het[-1, 0]])

if std_at_edge > 1.5 * std_at_center:
    print("✓ Uncertainty increases away from center")
else:
    print("❌ Uncertainty does NOT increase properly (heteroskedasticity not learned)")

# Check if uncertainty magnitude is reasonable
avg_ratio = np.mean([y_std_het[idx, 0] / noise_std[idx, 0] for idx in test_indices])
print(f"\nAverage learned/true uncertainty ratio: {avg_ratio:.2f}")
if 0.7 <= avg_ratio <= 1.3:
    print("✓ Uncertainty magnitude is reasonable")
elif avg_ratio < 0.7:
    print("❌ Uncertainty is UNDERESTIMATED")
else:
    print("❌ Uncertainty is OVERESTIMATED")
