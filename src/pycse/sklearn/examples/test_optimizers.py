"""Test script to verify DPOSE works with different optimizers."""

import jax
import numpy as np
from sklearn.model_selection import train_test_split
from dpose import DPOSE

# Generate simple test data
key = jax.random.PRNGKey(42)
x = np.linspace(0, 1, 100)[:, None]
noise_level = 0.01 + 0.05 * x.ravel()
y = x.ravel() ** 0.5 + noise_level * jax.random.normal(key, (100,))

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("Testing different optimizers with DPOSE\n")
print("=" * 70)

# Test each optimizer
optimizers = ["bfgs", "lbfgs", "adam", "gradient_descent"]

for opt in optimizers:
    print(f"\nTesting optimizer: {opt.upper()}")
    print("-" * 70)

    try:
        # Create model
        if opt in ["adam", "sgd"]:
            model = DPOSE(layers=(1, 10, 16), optimizer=opt, loss_type="crps")
            # Use fewer iterations and custom learning rate for iterative methods
            model.fit(x_train, y_train, val_X=x_val, val_y=y_val, maxiter=500, learning_rate=1e-3)
        else:
            model = DPOSE(layers=(1, 10, 16), optimizer=opt, loss_type="crps")
            model.fit(x_train, y_train, val_X=x_val, val_y=y_val)

        # Get predictions
        y_pred, y_std = model.predict(x_val, return_std=True)
        mae = np.mean(np.abs(y_val - y_pred))

        print(f"✓ {opt.upper()} succeeded")
        print(f"  MAE: {mae:.6f}")
        print(f"  Mean uncertainty: {np.mean(y_std):.6f}")

        # Show report
        model.report()

    except Exception as e:
        print(f"✗ {opt.upper()} failed: {e}")

print("\n" + "=" * 70)
print("Testing complete!")
