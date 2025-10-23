"""Test and benchmark Muon optimizer with DPOSE."""

import jax
import numpy as np
from sklearn.model_selection import train_test_split
from dpose import DPOSE

# Generate heteroscedastic test data
key = jax.random.PRNGKey(19)
x = np.linspace(0, 1, 200)[:, None]
noise_level = 0.01 + 0.08 * x.ravel()
y = x.ravel() ** (1 / 3) + noise_level * jax.random.normal(key, (200,))

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n" + "=" * 70)
print("DPOSE: Muon Optimizer Testing")
print("=" * 70)

# Test 1: Basic functionality
print("\n1. Testing Muon with default parameters")
print("-" * 70)
model_muon = DPOSE(layers=(1, 20, 32), optimizer="muon", loss_type="crps")
model_muon.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=500)
model_muon.report()

# Test 2: Custom parameters
print("\n2. Testing Muon with custom parameters")
print("-" * 70)
model_muon_custom = DPOSE(layers=(1, 20, 32), optimizer="muon", loss_type="crps")
model_muon_custom.fit(
    x_train,
    y_train,
    val_X=x_val,
    val_y=y_val,
    maxiter=500,
    learning_rate=0.03,
    beta=0.9,
    ns_steps=5,
)
model_muon_custom.report()

# Test 3: Compare with other optimizers
print("\n3. Performance Comparison")
print("-" * 70)

optimizers_to_test = [
    ("BFGS", "bfgs", {"maxiter": 500}),
    ("Adam", "adam", {"maxiter": 500, "learning_rate": 1e-3}),
    ("Muon", "muon", {"maxiter": 500, "learning_rate": 0.02}),
]

results = []

for name, opt, params in optimizers_to_test:
    print(f"\nTraining with {name}...", end=" ")
    model = DPOSE(layers=(1, 20, 32), optimizer=opt, loss_type="crps")
    model.fit(x_train, y_train, val_X=x_val, val_y=y_val, **params)

    y_pred, y_std = model.predict(x_val, return_std=True)
    mae = float(np.mean(np.abs(y_val - y_pred)))
    mean_std = float(np.mean(y_std))

    results.append(
        {
            "name": name,
            "mae": mae,
            "mean_std": mean_std,
            "calibration": model.calibration_factor,
            "iterations": model.state.iter_num if hasattr(model.state, "iter_num") else "N/A",
        }
    )
    print(f"Done (MAE: {mae:.6f})")

# Display comparison table
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)
print(f"{'Optimizer':<15} {'Iterations':<12} {'MAE':<12} {'Mean σ':<12} {'Calibration α':<15}")
print("-" * 70)

for r in results:
    iters = str(r["iterations"]) if r["iterations"] != "N/A" else "N/A"
    print(
        f"{r['name']:<15} {iters:<12} {r['mae']:<12.6f} {r['mean_std']:<12.6f} {r['calibration']:<15.4f}"
    )

print("\n" + "=" * 70)
print("Key Findings:")
print("  - Muon uses orthogonalized momentum for 2D parameters")
print("  - Typically requires higher learning rate (0.02 vs 0.001)")
print("  - State-of-the-art sample efficiency (Keller Jordan et al. 2024)")
print("  - Available in Optax contrib module")
print("=" * 70 + "\n")
