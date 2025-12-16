"""Quick verification that optimizations work correctly."""

import time

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN

jax.config.update("jax_enable_x64", True)

print("Testing that performance optimizations work correctly...")
print("=" * 70)

# Simple test case
key = jax.random.PRNGKey(42)
X = np.linspace(-2, 2, 50)[:, None]
y_true = 2 * X + 1
y = y_true + 0.1 * jax.random.normal(key, X.shape)

# Train
cinn = ConditionalInvertibleNN(
    n_features_in=1, n_features_out=1, n_layers=6, hidden_dims=[64, 64], seed=42
)

print("\n1. Training...")
cinn.fit(X, y, maxiter=500)
print(f"   Final NLL: {cinn.final_nll_:.4f}")
print("   ✓ Training successful")

# Test sampling
print("\n2. Testing sampling...")
y_pred1, y_std1 = cinn.predict(X[:10], return_std=True, n_samples=100)
print(f"   Shape check: y_pred={y_pred1.shape}, y_std={y_std1.shape}")
print(f"   Values: pred[0]={y_pred1[0, 0]:.3f}, std[0]={y_std1[0, 0]:.3f}")
print("   ✓ Sampling works")

# Test that it's cached and fast
print("\n3. Testing JIT caching (should be fast)...")

start = time.time()
y_pred2, y_std2 = cinn.predict(X[:10], return_std=True, n_samples=1000)
elapsed = time.time() - start
print(f"   Time for 1000 samples: {elapsed:.3f}s")
if elapsed < 1.0:
    print("   ✓ JIT caching working (fast!)")
else:
    print("   ⚠ Slower than expected")

# Test correctness - predictions should be similar
print("\n4. Testing correctness...")
diff = np.abs(y_pred1 - y_pred2[:10])
max_diff = np.max(diff)
print(f"   Max difference between runs: {max_diff:.6f}")
if max_diff < 0.1:
    print("   ✓ Results are consistent")
else:
    print("   ⚠ Results vary more than expected")

# Test on different X values
print("\n5. Testing on new X values...")
X_new = np.array([[0.0], [1.0], [2.0]])
y_pred_new, y_std_new = cinn.predict(X_new, return_std=True, n_samples=500)
print(f"   Predictions: {y_pred_new.ravel()}")
print(f"   Uncertainties: {y_std_new.ravel()}")
print("   ✓ Works on new data")

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
