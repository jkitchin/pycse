"""Test if initialization is working correctly."""

import jax
import jax.numpy as jnp
import numpy as np
from pycse.sklearn.dpose import DPOSE

# Generate simple test data
key = jax.random.PRNGKey(19)
x = np.linspace(0, 1, 50)[:, None]
y = x.ravel() + 0.1 * jax.random.normal(key, (50,))

print("Data types:")
print(f"  x type: {type(x)}, dtype: {x.dtype}")
print(f"  y type: {type(y)}, dtype: {y.dtype}")

# Create model
model = DPOSE(layers=(1, 10, 16), loss_type="nll", seed=19)

# Initialize parameters manually to check
print("\n" + "=" * 70)
print("Testing manual initialization")
print("=" * 70)

import jax.numpy as jnp_jax
from flax import linen as nn
from flax.linen.initializers import xavier_uniform


# Create network
class _NN(nn.Module):
    layers: tuple
    activation: callable

    @nn.compact
    def __call__(self, x):
        for i in self.layers[0:-1]:
            x = nn.Dense(i, kernel_init=xavier_uniform())(x)
            x = self.activation(x)
        x = nn.Dense(self.layers[-1])(x)
        return x


nn_test = _NN(layers=(1, 10, 16), activation=nn.relu)

# Try initialization with numpy array (user's approach)
print("\nInitializing with numpy array:")
try:
    params = nn_test.init(jax.random.PRNGKey(19), x[:5])  # First 5 samples
    print("  ✓ Initialization succeeded")
    print(f"  Params type: {type(params)}")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")

# Try initialization with jax array
print("\nInitializing with jax array:")
try:
    x_jax = jnp_jax.asarray(x[:5])
    params = nn_test.init(jax.random.PRNGKey(19), x_jax)
    print("  ✓ Initialization succeeded")
    print(f"  Params type: {type(params)}")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")

# Test forward pass
print("\n" + "=" * 70)
print("Testing forward pass")
print("=" * 70)

try:
    output = nn_test.apply(params, jnp_jax.asarray(x))
    print(f"✓ Forward pass succeeded")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Contains NaN: {jnp_jax.any(jnp_jax.isnan(output))}")
    print(f"  Contains inf: {jnp_jax.any(jnp_jax.isinf(output))}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")

# Test objective function calculation
print("\n" + "=" * 70)
print("Testing NLL objective")
print("=" * 70)

try:
    pY = nn_test.apply(params, jnp_jax.asarray(x))
    py = pY.mean(axis=1)
    sigma = pY.std(axis=1) + 1e-6
    errs = jnp_jax.asarray(y).ravel() - py

    print(f"Ensemble predictions shape: {pY.shape}")
    print(f"Mean prediction shape: {py.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Sigma range: [{sigma.min():.2e}, {sigma.max():.2e}]")
    print(f"Sigma contains NaN: {jnp_jax.any(jnp_jax.isnan(sigma))}")
    print(f"Errors range: [{errs.min():.4f}, {errs.max():.4f}]")

    nll = 0.5 * (errs**2 / sigma**2 + jnp_jax.log(2 * jnp_jax.pi * sigma**2))
    nll_mean = jnp_jax.mean(nll)

    print(f"\nNLL per sample range: [{nll.min():.4f}, {nll.max():.4f}]")
    print(f"Mean NLL: {nll_mean:.4f}")
    print(f"NLL contains NaN: {jnp_jax.any(jnp_jax.isnan(nll))}")
    print(f"NLL contains inf: {jnp_jax.any(jnp_jax.isinf(nll))}")

    if jnp_jax.isnan(nll_mean):
        print("\n✗ NLL is NaN!")
        print(f"  Checking individual terms:")
        print(f"    errs²/sigma²: NaN count = {jnp_jax.sum(jnp_jax.isnan(errs**2 / sigma**2))}")
        print(
            f"    log(2π sigma²): NaN count = {jnp_jax.sum(jnp_jax.isnan(jnp_jax.log(2 * jnp_jax.pi * sigma**2)))}"
        )
    else:
        print("\n✓ NLL calculation succeeded!")

except Exception as e:
    print(f"✗ Objective calculation failed: {e}")
    import traceback

    traceback.print_exc()

# Now try actual model fitting
print("\n" + "=" * 70)
print("Testing model.fit()")
print("=" * 70)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

try:
    model = DPOSE(layers=(1, 10, 16), loss_type="nll", seed=19)
    model.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=10)
    model.report()

    if jnp_jax.isnan(model.state.value):
        print("\n✗ Model fitting produced NaN loss!")
    else:
        print("\n✓ Model fitting succeeded!")

except Exception as e:
    print(f"✗ Model fitting failed: {e}")
    import traceback

    traceback.print_exc()
