"""Test if more training iterations fix the linear regression."""

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN

jax.config.update("jax_enable_x64", True)

print("=" * 70)
print("Testing Training Convergence")
print("=" * 70)

# Simple linear data
key = jax.random.PRNGKey(42)
X = np.linspace(-3, 3, 100)[:, None]
y_true = 2 * X + 1
y = y_true + 0.1 * jax.random.normal(key, X.shape)

# Test different numbers of iterations
test_configs = [
    (500, "Quick"),
    (1000, "Standard"),
    (2000, "Extended"),
    (3000, "Long"),
]

for maxiter, label in test_configs:
    print(f"\n{label} training ({maxiter} iterations)")
    print("-" * 70)

    cinn = ConditionalInvertibleNN(
        n_features_in=1, n_features_out=1, n_layers=8, hidden_dims=[128, 128], seed=42
    )

    cinn.fit(X, y, maxiter=maxiter)

    # Mode prediction
    y_pred_mode = cinn.predict(X)
    mse_mode = np.mean((y_pred_mode - y_true) ** 2)

    # Check if converged
    converged = cinn.state_.iter_num < cinn.maxiter

    print(f"  Final NLL: {cinn.final_nll_:.4f}")
    print(f"  Iterations used: {cinn.state_.iter_num}/{maxiter}")
    print(f"  Converged: {converged}")
    print(f"  Mode MSE: {mse_mode:.4f}")

    if mse_mode < 0.05:
        print("  ✓ Good fit!")
        break
    elif mse_mode < 1.0:
        print("  ⚠ Acceptable but not great")
    else:
        print("  ❌ Poor fit")

print("\n" + "=" * 70)
print("Conclusion:")
if mse_mode < 0.05:
    print(f"✓ Model works with {maxiter} iterations")
else:
    print("❌ Model struggles even with long training")
print("=" * 70)
