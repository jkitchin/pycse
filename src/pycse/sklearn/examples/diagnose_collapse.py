"""Diagnose why ensemble collapse is occurring."""

import jax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pycse.sklearn.dpose import DPOSE

# Generate heteroscedastic data (same as your code)
key = jax.random.PRNGKey(19)
x = np.linspace(0, 1, 100)[:, None]
noise_level = 0.01 + 0.1 * x.ravel()  # Increasing noise
y = x.ravel() ** (1 / 3) + noise_level * jax.random.normal(key, (100,))

# Split into train/validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("=" * 70)
print("DATA DIAGNOSTICS")
print("=" * 70)
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
print(f"y mean: {y.mean():.4f}, std: {y.std():.4f}")
print(f"Noise range: [{noise_level.min():.4f}, {noise_level.max():.4f}]")

# Test 1: Default training (1500 iterations)
print("\n" + "=" * 70)
print("TEST 1: Default Training (maxiter=1500)")
print("=" * 70)

model1 = DPOSE(layers=(1, 15, 32), loss_type="nll", seed=19)
model1.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=1500)
model1.report()

# Check ensemble spread on training data
ensemble1 = model1.predict_ensemble(x_train)
spread1 = ensemble1.std(axis=1)
print(f"\nEnsemble spread on training data:")
print(f"  Min: {spread1.min():.2e}")
print(f"  Mean: {spread1.mean():.2e}")
print(f"  Max: {spread1.max():.2e}")

if spread1.mean() < 1e-6:
    print("  ✗ COLLAPSED! Mean spread < 1e-6")
else:
    print("  ✓ Healthy diversity")

# Test 2: Fewer iterations
print("\n" + "=" * 70)
print("TEST 2: Reduced Iterations (maxiter=500)")
print("=" * 70)

model2 = DPOSE(layers=(1, 15, 32), loss_type="nll", seed=19)
model2.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=500)
model2.report()

ensemble2 = model2.predict_ensemble(x_train)
spread2 = ensemble2.std(axis=1)
print(f"\nEnsemble spread on training data:")
print(f"  Min: {spread2.min():.2e}")
print(f"  Mean: {spread2.mean():.2e}")
print(f"  Max: {spread2.max():.2e}")

if spread2.mean() < 1e-6:
    print("  ✗ COLLAPSED! Mean spread < 1e-6")
else:
    print("  ✓ Healthy diversity")

# Test 3: Even fewer iterations
print("\n" + "=" * 70)
print("TEST 3: Early Stop (maxiter=200)")
print("=" * 70)

model3 = DPOSE(layers=(1, 15, 32), loss_type="nll", seed=19)
model3.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=200)
model3.report()

ensemble3 = model3.predict_ensemble(x_train)
spread3 = ensemble3.std(axis=1)
print(f"\nEnsemble spread on training data:")
print(f"  Min: {spread3.min():.2e}")
print(f"  Mean: {spread3.mean():.2e}")
print(f"  Max: {spread3.max():.2e}")

if spread3.mean() < 1e-6:
    print("  ✗ COLLAPSED! Mean spread < 1e-6")
else:
    print("  ✓ Healthy diversity")

# Test 4: Larger ensemble
print("\n" + "=" * 70)
print("TEST 4: Larger Ensemble (64 members, maxiter=500)")
print("=" * 70)

model4 = DPOSE(layers=(1, 15, 64), loss_type="nll", seed=19)
model4.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=500)
model4.report()

ensemble4 = model4.predict_ensemble(x_train)
spread4 = ensemble4.std(axis=1)
print(f"\nEnsemble spread on training data:")
print(f"  Min: {spread4.min():.2e}")
print(f"  Mean: {spread4.mean():.2e}")
print(f"  Max: {spread4.max():.2e}")

if spread4.mean() < 1e-6:
    print("  ✗ COLLAPSED! Mean spread < 1e-6")
else:
    print("  ✓ Healthy diversity")

# Visualization comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = [model1, model2, model3, model4]
titles = [
    "Default (1500 iter)",
    "Reduced (500 iter)",
    "Early (200 iter)",
    "Large Ensemble (64, 500 iter)",
]
spreads = [spread1, spread2, spread3, spread4]

for idx, (ax, model, title, spread) in enumerate(zip(axes.flat, models, titles, spreads)):
    # Get predictions
    y_pred, y_std = model.predict(x, return_std=True)

    # Sort for plotting
    sort_idx = np.argsort(x.ravel())
    x_sorted = x.ravel()[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_std_sorted = y_std[sort_idx]

    # Plot
    ax.fill_between(
        x_sorted,
        y_pred_sorted - 2 * y_std_sorted,
        y_pred_sorted + 2 * y_std_sorted,
        alpha=0.3,
        color="red",
    )
    ax.plot(x_sorted, y_pred_sorted, "r-", linewidth=2, label="Mean prediction")
    ax.scatter(x, y, alpha=0.5, s=20, color="blue", label="Data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}\nMean spread: {spread.mean():.2e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("collapse_diagnosis.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved diagnostic visualization to 'collapse_diagnosis.png'")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(
    f"Test 1 (1500 iter): {'COLLAPSED' if spread1.mean() < 1e-6 else 'OK'} - spread = {spread1.mean():.2e}"
)
print(
    f"Test 2 (500 iter):  {'COLLAPSED' if spread2.mean() < 1e-6 else 'OK'} - spread = {spread2.mean():.2e}"
)
print(
    f"Test 3 (200 iter):  {'COLLAPSED' if spread3.mean() < 1e-6 else 'OK'} - spread = {spread3.mean():.2e}"
)
print(
    f"Test 4 (64 members): {'COLLAPSED' if spread4.mean() < 1e-6 else 'OK'} - spread = {spread4.mean():.2e}"
)

print("\nRECOMMENDATION:")
if spread2.mean() >= 1e-6:
    print("✓ Use maxiter=500 instead of default 1500")
    print("\n  model = DPOSE(layers=(1, 15, 32), loss_type='nll')")
    print("  model.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=500)")
elif spread3.mean() >= 1e-6:
    print("✓ Use maxiter=200 to prevent overfitting")
    print("\n  model = DPOSE(layers=(1, 15, 32), loss_type='nll')")
    print("  model.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=200)")
elif spread4.mean() >= 1e-6:
    print("✓ Use larger ensemble (64 members) with maxiter=500")
    print("\n  model = DPOSE(layers=(1, 15, 64), loss_type='nll')")
    print("  model.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=500)")
else:
    print("⚠ All tests collapsed - may need different approach:")
    print("  - Normalize/scale the data")
    print("  - Use CRPS loss instead of NLL")
    print("  - Add L2 regularization")

print("=" * 70)
