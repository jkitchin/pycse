"""Examples of using different optimizers with DPOSE.

This script demonstrates how to use various optimization algorithms
with the DPOSE neural network model.
"""

import jax
import numpy as np
from sklearn.model_selection import train_test_split
from dpose import DPOSE

# Generate heteroscedastic test data
key = jax.random.PRNGKey(42)
x = np.linspace(0, 1, 200)[:, None]
noise_level = 0.01 + 0.08 * x.ravel()
y = x.ravel() ** (1 / 3) + noise_level * jax.random.normal(key, (200,))

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n" + "=" * 70)
print("DPOSE: Examples of Different Optimizers")
print("=" * 70)

# Example 1: BFGS (default, recommended for most cases)
print("\n1. BFGS (Default - Recommended)")
print("-" * 70)
model_bfgs = DPOSE(layers=(1, 20, 32), optimizer="bfgs", loss_type="crps")
model_bfgs.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=1500)
model_bfgs.report()

# Example 2: L-BFGS (good for larger problems)
print("\n2. L-BFGS (Memory-efficient for large problems)")
print("-" * 70)
model_lbfgs = DPOSE(layers=(1, 20, 32), optimizer="lbfgs", loss_type="crps")
model_lbfgs.fit(x_train, y_train, val_X=x_val, val_y=y_val)
model_lbfgs.report()

# Example 3: Adam (adaptive learning rate)
print("\n3. Adam (Good for deep networks)")
print("-" * 70)
model_adam = DPOSE(layers=(1, 20, 32), optimizer="adam", loss_type="crps")
model_adam.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=1000, learning_rate=1e-3)
model_adam.report()

# Example 4: SGD with momentum
print("\n4. SGD with Momentum")
print("-" * 70)
model_sgd = DPOSE(layers=(1, 20, 32), optimizer="sgd", loss_type="crps")
model_sgd.fit(
    x_train, y_train, val_X=x_val, val_y=y_val, maxiter=1000, learning_rate=1e-2, momentum=0.9
)
model_sgd.report()

# Example 5: Gradient Descent (basic)
print("\n5. Gradient Descent (Basic)")
print("-" * 70)
model_gd = DPOSE(layers=(1, 20, 32), optimizer="gradient_descent", loss_type="crps")
model_gd.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=500)
model_gd.report()

# Example 6: Muon (state-of-the-art 2024)
print("\n6. Muon (State-of-the-art 2024)")
print("-" * 70)
model_muon = DPOSE(layers=(1, 20, 32), optimizer="muon", loss_type="crps")
model_muon.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=500, learning_rate=0.02)
model_muon.report()

# Compare performance
print("\n" + "=" * 70)
print("Performance Comparison on Validation Set")
print("=" * 70)

models = {
    "BFGS": model_bfgs,
    "L-BFGS": model_lbfgs,
    "Adam": model_adam,
    "SGD": model_sgd,
    "Gradient Descent": model_gd,
    "Muon": model_muon,
}

print(f"\n{'Optimizer':<20} {'MAE':<12} {'Mean σ':<12} {'Calibration α':<12}")
print("-" * 70)

for name, model in models.items():
    y_pred, y_std = model.predict(x_val, return_std=True)
    mae = float(np.mean(np.abs(y_val - y_pred)))
    mean_std = float(np.mean(y_std))
    calib = model.calibration_factor

    print(f"{name:<20} {mae:<12.6f} {mean_std:<12.6f} {calib:<12.4f}")

print("\n" + "=" * 70)
print("Recommendations:")
print("  - BFGS: Best overall, fastest convergence for small/medium networks")
print("  - L-BFGS: Use for larger networks when memory is a concern")
print("  - Adam: Good for deep networks, requires tuning learning_rate")
print("  - SGD: Classical choice, may need learning rate scheduling")
print("  - Muon: State-of-the-art (2024), best sample efficiency, <3% overhead")
print("  - Gradient Descent: Simple baseline, usually slower")
print("=" * 70 + "\n")
