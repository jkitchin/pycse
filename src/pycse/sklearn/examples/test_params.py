"""Check actual parameter initialization values."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import xavier_uniform


# Network definition from dpose.py
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


# Create network
network = _NN(layers=(1, 10, 16), activation=nn.relu)

# Initialize
key = jax.random.PRNGKey(19)
x_dummy = jnp.ones((5, 1))  # 5 samples, 1 feature

params = network.init(key, x_dummy)

print("=" * 70)
print("PARAMETER INSPECTION")
print("=" * 70)

# Check parameter structure
print("\nParameter structure:")
print(params.keys())

# Inspect each layer
for layer_name in params["params"].keys():
    print(f"\n{layer_name}:")
    layer_params = params["params"][layer_name]

    if "kernel" in layer_params:
        kernel = layer_params["kernel"]
        print(f"  Kernel shape: {kernel.shape}")
        print(f"  Kernel mean: {jnp.mean(kernel):.6f}")
        print(f"  Kernel std: {jnp.std(kernel):.6f}")
        print(f"  Kernel range: [{jnp.min(kernel):.6f}, {jnp.max(kernel):.6f}]")
        print(f"  All zeros: {jnp.all(kernel == 0)}")

    if "bias" in layer_params:
        bias = layer_params["bias"]
        print(f"  Bias shape: {bias.shape}")
        print(f"  Bias mean: {jnp.mean(bias):.6f}")
        print(f"  Bias std: {jnp.std(bias):.6f}")
        print(f"  Bias range: [{jnp.min(bias):.6f}, {jnp.max(bias):.6f}]")
        print(f"  All zeros: {jnp.all(bias == 0)}")

# Test forward pass step by step
print("\n" + "=" * 70)
print("FORWARD PASS STEP-BY-STEP")
print("=" * 70)

x_test = jnp.array([[0.5], [0.7], [0.9]])  # 3 samples
print(f"\nInput: {x_test.ravel()}")

# Manual forward pass
x = x_test

# Layer 0 (Dense with 10 neurons + ReLU)
kernel_0 = params["params"]["Dense_0"]["kernel"]
bias_0 = params["params"]["Dense_0"]["bias"]
x = jnp.dot(x, kernel_0) + bias_0
print(f"\nAfter Dense_0 (before ReLU):")
print(f"  Range: [{x.min():.6f}, {x.max():.6f}]")
print(f"  Mean: {x.mean():.6f}")

x = nn.relu(x)
print(f"After ReLU:")
print(f"  Range: [{x.min():.6f}, {x.max():.6f}]")
print(f"  Mean: {x.mean():.6f}")
print(f"  All zeros: {jnp.all(x == 0)}")

# Layer 1 (Dense with 16 outputs, no activation)
kernel_1 = params["params"]["Dense_1"]["kernel"]
bias_1 = params["params"]["Dense_1"]["bias"]
x = jnp.dot(x, kernel_1) + bias_1
print(f"\nAfter Dense_1 (final output):")
print(f"  Range: [{x.min():.6f}, {x.max():.6f}]")
print(f"  Mean: {x.mean():.6f}")
print(f"  All zeros: {jnp.all(x == 0)}")

# Compare with network apply
output_apply = network.apply(params, x_test)
print(f"\nUsing network.apply:")
print(f"  Range: [{output_apply.min():.6f}, {output_apply.max():.6f}]")
print(f"  Mean: {output_apply.mean():.6f}")

# Test with different initialization seeds
print("\n" + "=" * 70)
print("TESTING DIFFERENT SEEDS")
print("=" * 70)

for seed in [0, 1, 19, 42, 100]:
    key = jax.random.PRNGKey(seed)
    params_test = network.init(key, x_dummy)
    output_test = network.apply(params_test, x_test)

    print(f"\nSeed {seed}:")
    print(f"  Output range: [{output_test.min():.6f}, {output_test.max():.6f}]")
    print(f"  Output mean: {output_test.mean():.6f}")
    print(f"  All zeros: {jnp.all(output_test == 0)}")
