"""Test that Conditional INN correctly uses X context to learn p(Y|X)."""

import jax
import jax.numpy as np
from pycse.sklearn.cinn import ConditionalInvertibleNN

# Enable 64-bit
import os

os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


def test_cinn_learns_conditional_distribution():
    """Test that CINN learns p(Y|X) not p(Y)."""

    # Create simple conditional data: y = 2*x + noise
    key = jax.random.PRNGKey(42)
    n_train = 500

    key, subkey = jax.random.split(key)
    X_train = jax.random.uniform(subkey, (n_train, 1), minval=-2, maxval=2)
    y_train = 2.0 * X_train + 0.1 * jax.random.normal(key, (n_train, 1))

    # Train model
    cinn = ConditionalInvertibleNN(
        n_features_in=1, n_features_out=1, n_layers=4, hidden_dims=[32, 32], seed=42
    )
    cinn.fit(X_train, y_train, maxiter=1000)

    # Test: Sample from p(Y|X=0) and p(Y|X=1)
    # These should be DIFFERENT distributions
    X_test_0 = np.array([[0.0]])
    X_test_1 = np.array([[1.0]])

    key, subkey = jax.random.split(key)
    samples_at_0 = cinn.sample(X_test_0, n_samples=1000, key=subkey)

    key, subkey = jax.random.split(key)
    samples_at_1 = cinn.sample(X_test_1, n_samples=1000, key=subkey)

    # Compute means
    mean_at_0 = float(np.mean(samples_at_0))
    mean_at_1 = float(np.mean(samples_at_1))

    # Expected: mean_at_0 ~ 0.0, mean_at_1 ~ 2.0, difference ~ 2.0
    assert abs(mean_at_0 - 0.0) < 0.3, f"p(Y|X=0) mean should be ~0, got {mean_at_0}"
    assert abs(mean_at_1 - 2.0) < 0.3, f"p(Y|X=1) mean should be ~2, got {mean_at_1}"
    assert abs(mean_at_1 - mean_at_0 - 2.0) < 0.5, (
        f"Difference should be ~2.0, got {abs(mean_at_1 - mean_at_0)}"
    )


def test_cinn_forward_depends_on_context():
    """Test that forward pass depends on X context."""

    cinn = ConditionalInvertibleNN(
        n_features_in=1, n_features_out=1, n_layers=2, hidden_dims=[16], seed=42
    )

    # Create dummy training data to fit the model
    key = jax.random.PRNGKey(42)
    X_dummy = jax.random.normal(key, (100, 1))
    y_dummy = jax.random.normal(key, (100, 1))
    cinn.fit(X_dummy, y_dummy, maxiter=10)

    # Test forward pass with different X
    X_test_0 = np.array([[0.0]])
    X_test_1 = np.array([[1.0]])
    y_test = np.array([[0.5]])

    # Normalize and pad
    X_0_norm = (X_test_0 - cinn.X_mean_) / cinn.X_std_
    X_1_norm = (X_test_1 - cinn.X_mean_) / cinn.X_std_
    y_norm = (y_test - cinn.y_mean_) / cinn.y_std_
    y_padded = cinn._pad_1d_output(y_norm)

    # Forward pass
    z0, _ = cinn.flow.apply(cinn.params_, y_padded, X_0_norm)
    z1, _ = cinn.flow.apply(cinn.params_, y_padded, X_1_norm)

    # z should depend on X
    assert np.abs(z1 - z0).max() > 1e-6, "Forward pass should depend on X context"


if __name__ == "__main__":
    test_cinn_learns_conditional_distribution()
    test_cinn_forward_depends_on_context()
    print("✓ All tests passed!")
