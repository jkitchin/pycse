"""Demo of ActiveSurrogate for automatic surrogate modeling."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from pycse.pyroxy import ActiveSurrogate


# Define an expensive function to surrogate
def expensive_function(X):
    """A moderately complex 1D function."""
    x = X.flatten()
    return np.sin(x) + 0.5 * np.sin(3 * x) + 0.1 * x


# Define domain
bounds = [(0, 10)]

# Create model
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

# Build surrogate with active learning
print("Building surrogate with active learning...")
surrogate, history = ActiveSurrogate.build(
    func=expensive_function,
    bounds=bounds,
    model=model,
    acquisition="ei",
    stopping_criterion="mean_ratio",
    stopping_threshold=1.5,
    n_initial=5,
    max_iterations=30,
    verbose=True,
)

print(f"\nFinal model: {len(surrogate.xtrain)} samples")

# Visualize results
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
y_true = expensive_function(X_plot)
y_pred = surrogate(X_plot)

plt.figure(figsize=(12, 8))

# Plot 1: Function and surrogate
plt.subplot(2, 2, 1)
plt.plot(X_plot, y_true, "b-", label="True function", linewidth=2)
plt.plot(X_plot, y_pred, "r--", label="Surrogate", linewidth=2)
plt.scatter(
    surrogate.xtrain,
    surrogate.ytrain,
    c="black",
    s=50,
    label=f"Samples (n={len(surrogate.xtrain)})",
    zorder=5,
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Surrogate vs True Function")
plt.grid(True, alpha=0.3)

# Plot 2: Sample progression
plt.subplot(2, 2, 2)
plt.plot(history["iterations"], history["n_samples"], "b-o")
plt.xlabel("Iteration")
plt.ylabel("Total Samples")
plt.title("Sample Count vs Iteration")
plt.grid(True, alpha=0.3)

# Plot 3: Uncertainty evolution
plt.subplot(2, 2, 3)
plt.plot(history["iterations"], history["mean_uncertainty"], "g-o", label="Mean")
plt.plot(history["iterations"], history["max_uncertainty"], "r-s", label="Max")
plt.xlabel("Iteration")
plt.ylabel("Uncertainty")
plt.legend()
plt.title("Uncertainty Evolution")
plt.grid(True, alpha=0.3)

# Plot 4: Acquisition values
plt.subplot(2, 2, 4)
plt.plot(history["iterations"], history["acquisition_values"], "m-o")
plt.xlabel("Iteration")
plt.ylabel("Best Acquisition Value")
plt.title("Acquisition Value vs Iteration")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("active_surrogate_demo.png", dpi=150)
print("\nVisualization saved to active_surrogate_demo.png")
plt.show()
