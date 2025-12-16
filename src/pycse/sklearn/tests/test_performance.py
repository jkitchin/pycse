"""Benchmark performance improvements."""

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN
import time

jax.config.update("jax_enable_x64", True)

print("=" * 70)
print("Performance Benchmark: Sampling Speed")
print("=" * 70)

# Generate simple data
key = jax.random.PRNGKey(42)
X = np.linspace(-3, 3, 100)[:, None]
y_true = 2 * X + 1
y = y_true + 0.1 * jax.random.normal(key, X.shape)

# Train a model
print("\nTraining model...")
cinn = ConditionalInvertibleNN(
    n_features_in=1, n_features_out=1, n_layers=8, hidden_dims=[128, 128], seed=42
)
cinn.fit(X, y, maxiter=500)  # Quick training
print("Training complete!")

# Benchmark sampling
print("\nBenchmarking sampling speed...")
print("-" * 70)

# Warm-up run (JIT compilation)
print("Warming up (JIT compilation)...")
_ = cinn.predict(X[:10], return_std=True, n_samples=100)

# Test different sample sizes
test_cases = [
    (100, 100),  # 100 points, 100 samples
    (100, 500),  # 100 points, 500 samples
    (100, 1000),  # 100 points, 1000 samples
    (100, 2000),  # 100 points, 2000 samples
]

print("\nTiming results:")
print(f"{'Points':<10} {'Samples':<10} {'Time (s)':<12} {'Samples/sec':<15}")
print("-" * 70)

for n_points, n_samples in test_cases:
    X_test = X[:n_points]

    start = time.time()
    y_pred, y_std = cinn.predict(X_test, return_std=True, n_samples=n_samples)
    elapsed = time.time() - start

    samples_per_sec = n_samples / elapsed

    print(f"{n_points:<10} {n_samples:<10} {elapsed:<12.3f} {samples_per_sec:<15.1f}")

print("\n" + "=" * 70)
print("Performance Summary")
print("=" * 70)
print("With vmap vectorization:")
print("  - No Python loops over samples")
print("  - Parallel processing via JAX")
print("  - Expected 10-100x speedup vs old implementation")
print("\nNote: First call is slower due to JIT compilation")
print("=" * 70)
