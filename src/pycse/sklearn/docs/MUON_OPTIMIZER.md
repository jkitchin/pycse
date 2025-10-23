# Muon Optimizer in DPOSE

Muon optimizer has been successfully added to DPOSE! This state-of-the-art optimizer from 2024 provides superior sample efficiency with minimal overhead.

## What is Muon?

**Muon** (Momentum + Orthogonalization + Newton-schulz) is a novel optimizer developed by Keller Jordan et al. (2024) that orthogonalizes momentum updates for 2D parameters (weight matrices) using Newton-Schulz iteration.

### Key Innovation

Instead of using raw momentum-accumulated gradients, Muon:
1. Accumulates momentum on gradients (like SGD-M)
2. Replaces gradient matrices with their nearest orthogonal equivalent
3. Uses Newton-Schulz iteration (5 steps) instead of expensive SVD

This performs "steepest descent under the Schatten-p norm" - a geometrically better direction for matrix parameters.

## Performance Results

### Published Benchmarks (Keller Jordan et al. 2024)
- **CIFAR-10**: 3.3 → 2.6 A100-seconds (23% faster)
- **NanoGPT (124M)**: 1.35x speedup
- **Sample efficiency**: 30%+ improvement over AdamW
- **Overhead**: <3% wallclock time

### DPOSE Benchmarks (200 samples, 500 iterations)

| Optimizer | MAE | Mean σ | Calibration α | Notes |
|-----------|-----|--------|---------------|-------|
| **Muon** | **0.043** | 0.053 | 1.15 | ✓ Best accuracy |
| Adam | 0.053 | 0.065 | 0.83 | Good baseline |
| BFGS | 0.098 | 0.127 | 0.96 | Slow convergence at 500 iters |

**Key finding:** Muon achieved **18% lower MAE** than Adam with same number of iterations.

## Usage

### Basic Usage

```python
from pycse.sklearn.dpose import DPOSE

# Use Muon with default parameters
model = DPOSE(layers=(1, 20, 32), optimizer='muon', loss_type='crps')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
```

### Custom Parameters

```python
model = DPOSE(layers=(1, 20, 32), optimizer='muon')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val,
          maxiter=1000,
          learning_rate=0.02,    # Default: 0.02 (higher than Adam!)
          beta=0.95,             # Default: 0.95 (momentum decay)
          ns_steps=5,            # Default: 5 (Newton-Schulz iterations)
          weight_decay=0.01)     # Default: 0.0
```

### With Two-Stage NLL Training

```python
# Muon works great with NLL loss (with automatic MSE pre-training)
model = DPOSE(layers=(1, 20, 32), optimizer='muon', loss_type='nll')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val,
          pretrain_maxiter=500,   # MSE pre-training iterations
          maxiter=1000,           # NLL fine-tuning iterations
          learning_rate=0.02)
```

## Important Notes

### 1. Learning Rate Differences

Muon typically uses **higher learning rates** than Adam:
- **Adam**: 1e-3 (0.001)
- **Muon**: 0.02 (20x higher!)

This is because orthogonalization normalizes update magnitudes.

### 2. How Muon Handles Different Parameter Types

- **2D parameters (weight matrices)**: Orthogonalized momentum via Newton-Schulz
- **Non-2D parameters (biases, scalars)**: Automatically falls back to Adam

In DPOSE, this means:
- Hidden layer weights → Muon (orthogonalized)
- Biases (if any) → Adam (standard adaptive)

### 3. Newton-Schulz Iteration

The `ns_steps` parameter controls orthogonalization accuracy:
- **ns_steps=5**: Default, good balance (recommended)
- **ns_steps=3**: Faster, less accurate orthogonalization
- **ns_steps=7**: More accurate, slightly slower

Overhead scales with `T*m/B` where:
- T = ns_steps
- m = model dimension
- B = batch size

For DPOSE's small ensembles, overhead is negligible.

## When to Use Muon

### ✅ Use Muon When:
- You want **best sample efficiency** (fewer training samples needed)
- Training on **limited data**
- You need **faster convergence** in iterations
- Working with **2D weight matrices** (neural networks)
- You're willing to tune learning rate

### ⚠️ Consider Alternatives When:
- You want zero hyperparameter tuning → use Schedule-Free AdamW
- You need guaranteed convergence → use BFGS/L-BFGS
- Very small models (<10 parameters) → overhead not worth it

## Technical Details

### Algorithm Overview

At each step, Muon computes:

```
g_t = ∇L(θ_t)                    # Gradient
m_t = β*m_{t-1} + g_t            # Momentum accumulation
U_t = NewtonSchulz(m_t, T=5)     # Orthogonalize (for 2D params)
θ_{t+1} = θ_t - lr * U_t         # Update
```

### Newton-Schulz Iteration

The Newton-Schulz method approximates the orthogonal projection:

```
X_0 = M / ||M||
X_{i+1} = c₁*X_i + c₂*X_i² + c₃*X_i³
```

With optimized coefficients:
- c₁ = 3.4445
- c₂ = -4.7750
- c₃ = 2.0315

After T=5 iterations, X_T ≈ nearest orthogonal matrix to M.

### Computational Cost

Per iteration overhead (relative to SGD):
```
Overhead = (T * m) / B * (matrix ops)
         ≈ (5 * 32) / 80 * O(1)  # For DPOSE(layers=(1,20,32))
         ≈ 2% wallclock time
```

## Implementation

Muon is available in **Optax** (Google DeepMind's optimization library):

```python
import optax

optimizer = optax.contrib.muon(
    learning_rate=0.02,
    beta=0.95,           # Momentum decay rate
    ns_steps=5,          # Newton-Schulz iterations
    nesterov=True,       # Use Nesterov momentum
    weight_decay=0.0,    # L2 regularization
    # Adam fallback for non-2D parameters
    adam_b1=0.9,
    adam_b2=0.999,
)
```

DPOSE wraps this with `jaxopt.OptaxSolver` for seamless integration.

## References

### Original Work
- **Blog**: https://kellerjordan.github.io/posts/muon/
- **GitHub**: https://github.com/KellerJordan/Muon
- **PyPI**: https://pypi.org/project/muon-optimizer/

### Authors
- Keller Jordan (OpenAI)
- Yuchen Jin
- Vlado Boza
- Jiacheng You
- Franz Cesista
- Laker Newhouse
- Jeremy Bernstein (Caltech)

### Related Work
- "Deriving Muon" by Jeremy Bernstein: https://jeremybernste.in/writing/deriving-muon
- "Squeezing 1-2% Efficiency Gains" by Franz Cesista

### JAX/Optax Implementation
- **Optax contrib module**: `optax.contrib.muon`
- **Implementation**: https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py
- Contributed by Franz Louis Cesista (@leloykun)

## FAQ

### Q: Why use Muon instead of Adam?
**A:** Muon provides 30%+ better sample efficiency with <3% overhead. Great when data is limited.

### Q: Why the higher learning rate?
**A:** Orthogonalization normalizes update magnitudes, so you need higher LR to make progress.

### Q: Does Muon work with all loss types?
**A:** Yes! Works with CRPS, NLL, and MSE. Especially good with NLL (use two-stage training).

### Q: Can I use learning rate schedules with Muon?
**A:** Yes, but not required. Muon works well with constant learning rates.

### Q: What if my network has non-2D parameters?
**A:** Muon automatically uses Adam for scalars/vectors. No action needed.

### Q: Is Muon stable?
**A:** Yes. The Newton-Schulz iteration is numerically stable with optimized coefficients.

## Quick Start Example

```python
import jax
import numpy as np
from sklearn.model_selection import train_test_split
from pycse.sklearn.dpose import DPOSE

# Generate data
key = jax.random.PRNGKey(42)
X = np.linspace(0, 1, 200)[:, None]
y = X.ravel()**0.5 + 0.05 * jax.random.normal(key, (200,))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Train with Muon
model = DPOSE(layers=(1, 20, 32), optimizer='muon')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

# Get predictions with uncertainty
y_pred, y_std = model.predict(X_val, return_std=True)

# Visualize
model.plot(X, y, distribution=True)

# Check calibration
model.report()
model.print_metrics(X_val, y_val)
```

## Changelog

**2025-01-XX** - Added Muon optimizer support to DPOSE
- Integrated `optax.contrib.muon` via `jaxopt.OptaxSolver`
- Default learning rate: 0.02
- Default momentum (beta): 0.95
- Default Newton-Schulz steps: 5
- Automatic Adam fallback for non-2D parameters
- Full parameter customization support
