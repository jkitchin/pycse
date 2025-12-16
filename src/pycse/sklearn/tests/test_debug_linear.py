"""Debug what's wrong with linear regression."""

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN

jax.config.update("jax_enable_x64", True)

print("=" * 70)
print("Debugging Linear Regression Issue")
print("=" * 70)

# Simple linear data
key = jax.random.PRNGKey(42)
X = np.linspace(-3, 3, 100)[:, None]
y_true = 2 * X + 1
y = y_true + 0.1 * jax.random.normal(key, X.shape)

print("\nData range:")
print(f"  X: [{X.min():.2f}, {X.max():.2f}]")
print(f"  y: [{y.min():.2f}, {y.max():.2f}]")
print("  y_true: y = 2*X + 1")

# Train
cinn = ConditionalInvertibleNN(
    n_features_in=1, n_features_out=1, n_layers=8, hidden_dims=[128, 128], seed=42
)

print("\nTraining...")
cinn.fit(X, y, maxiter=1000)
print(f"Final NLL: {cinn.final_nll_:.4f}")

# Check normalization
print("\nNormalization parameters:")
print(f"  X_mean: {cinn.X_mean_}")
print(f"  X_std: {cinn.X_std_}")
print(f"  y_mean: {cinn.y_mean_}")
print(f"  y_std: {cinn.y_std_}")

# Predict without uncertainty first
print("\nPredicting without uncertainty (mode)...")
y_pred_mode = cinn.predict(X)
mse_mode = np.mean((y_pred_mode - y_true) ** 2)
print(f"Mode prediction MSE: {mse_mode:.4f}")

# Check a few specific predictions
test_x = np.array([[-2.0], [0.0], [2.0]])
test_y_true = 2 * test_x + 1
test_y_pred = cinn.predict(test_x)

print("\nTest predictions (mode):")
print(f"  X=-2: true={test_y_true[0, 0]:.2f}, pred={test_y_pred[0, 0]:.2f}")
print(f"  X=0:  true={test_y_true[1, 0]:.2f}, pred={test_y_pred[1, 0]:.2f}")
print(f"  X=2:  true={test_y_true[2, 0]:.2f}, pred={test_y_pred[2, 0]:.2f}")

# Now try with uncertainty (small sample)
print("\nPredicting with uncertainty (10 samples)...")
y_pred_unc, y_std = cinn.predict(test_x, return_std=True, n_samples=10)
print("Test predictions (with uncertainty):")
print(f"  X=-2: mean={y_pred_unc[0, 0]:.2f}, std={y_std[0, 0]:.2f}")
print(f"  X=0:  mean={y_pred_unc[1, 0]:.2f}, std={y_std[1, 0]:.2f}")
print(f"  X=2:  mean={y_pred_unc[2, 0]:.2f}, std={y_std[2, 0]:.2f}")

# Try larger sample
print("\nPredicting with more samples (100 samples)...")
y_pred_unc2, y_std2 = cinn.predict(test_x, return_std=True, n_samples=100)
print("Test predictions (100 samples):")
print(f"  X=-2: mean={y_pred_unc2[0, 0]:.2f}, std={y_std2[0, 0]:.2f}")
print(f"  X=0:  mean={y_pred_unc2[1, 0]:.2f}, std={y_std2[1, 0]:.2f}")
print(f"  X=2:  mean={y_pred_unc2[2, 0]:.2f}, std={y_std2[2, 0]:.2f}")

# Full prediction with samples
print("\nFull prediction on training data (100 samples)...")
y_pred_full, y_std_full = cinn.predict(X, return_std=True, n_samples=100)
mse_full = np.mean((y_pred_full - y_true) ** 2)
print(f"MSE with sampling: {mse_full:.4f}")
print(f"Mean prediction: [{y_pred_full.min():.2f}, {y_pred_full.max():.2f}]")
print(f"True range: [{y_true.min():.2f}, {y_true.max():.2f}]")

if mse_full > 10:
    print("\n❌ SOMETHING IS VERY WRONG!")
    print("Predictions are way off!")
else:
    print("\n✓ Predictions look reasonable")
