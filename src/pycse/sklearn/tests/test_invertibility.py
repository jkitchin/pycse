"""Test if forward/inverse are truly inverse of each other."""

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN

jax.config.update("jax_enable_x64", True)

print("=" * 70)
print("Testing Invertibility")
print("=" * 70)

# Simple data
key = jax.random.PRNGKey(42)
X = np.linspace(-3, 3, 20)[:, None]
y_true = 2 * X + 1
y = y_true + 0.1 * jax.random.normal(key, X.shape)

# Train
cinn = ConditionalInvertibleNN(
    n_features_in=1,
    n_features_out=1,
    n_layers=4,  # Fewer layers for faster testing
    hidden_dims=[64, 64],
    seed=42,
)

print("Training...")
cinn.fit(X, y, maxiter=500)
print(f"Final NLL: {cinn.final_nll_:.4f}")

# Test invertibility: y → z → y_reconstructed
print("\nTesting invertibility: y → forward → inverse → y_reconstructed")
print("-" * 70)

# Take a few test points
X_test = X[:5]
y_test = y[:5]

# Normalize
X_norm = (X_test - cinn.X_mean_) / cinn.X_std_
y_norm = (y_test - cinn.y_mean_) / cinn.y_std_

# Pad
y_norm_padded = cinn._pad_1d_output(y_norm)

print(f"\nOriginal y_norm_padded shape: {y_norm_padded.shape}")
print(f"First value (padded): {y_norm_padded[0]}")

# Forward: y → z
z, log_det = cinn.flow.apply(cinn.params_, y_norm_padded, X_norm, inverse=False)
print("\nForward transform (y → z):")
print(f"  z shape: {z.shape}")
print(f"  z[0]: {z[0]}")
print(f"  log_det[0]: {log_det[0]:.4f}")

# Inverse: z → y
y_reconstructed_padded = cinn.flow.apply(cinn.params_, z, X_norm, inverse=True)
print("\nInverse transform (z → y):")
print(f"  y_reconstructed shape: {y_reconstructed_padded.shape}")
print(f"  y_reconstructed[0]: {y_reconstructed_padded[0]}")

# Check reconstruction error
reconstruction_error = np.max(np.abs(y_norm_padded - y_reconstructed_padded))
print(f"\nReconstruction error: {reconstruction_error:.2e}")

if reconstruction_error < 1e-6:
    print("✓ Perfect invertibility!")
elif reconstruction_error < 1e-3:
    print("✓ Good invertibility (small numerical error)")
else:
    print(f"❌ INVERTIBILITY BROKEN! Error = {reconstruction_error:.2e}")
    print("\nDifference:")
    print(f"  Original:       {y_norm_padded[0]}")
    print(f"  Reconstructed:  {y_reconstructed_padded[0]}")
    print(f"  Difference:     {y_norm_padded[0] - y_reconstructed_padded[0]}")

# Also check the unpadded version
y_reconstructed_unpadded = cinn._unpad_1d_output(y_reconstructed_padded)
y_reconstructed_denorm = y_reconstructed_unpadded * cinn.y_std_ + cinn.y_mean_

print("\nFinal check:")
print(f"  Original y:       {y_test[0, 0]:.4f}")
print(f"  Reconstructed y:  {y_reconstructed_denorm[0, 0]:.4f}")
print(f"  Difference:       {abs(y_test[0, 0] - y_reconstructed_denorm[0, 0]):.2e}")
