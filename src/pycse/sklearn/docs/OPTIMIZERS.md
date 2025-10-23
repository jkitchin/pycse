# DPOSE Optimizers Guide

Complete guide to optimization algorithms available in DPOSE.

## Quick Start

```python
from pycse.sklearn.dpose import DPOSE

# Default (BFGS - recommended for most cases)
model = DPOSE(layers=(1, 20, 32))
model.fit(X, y)

# Muon (state-of-the-art 2024)
model = DPOSE(layers=(1, 20, 32), optimizer='muon')
model.fit(X, y, learning_rate=0.02)

# Adam (deep networks)
model = DPOSE(layers=(1, 20, 32), optimizer='adam')
model.fit(X, y, learning_rate=1e-3)
```

## Available Optimizers

### 1. BFGS (Default)

**Best for:** Small to medium networks, guaranteed convergence

```python
model = DPOSE(layers=(1, 20, 32), optimizer='bfgs')
model.fit(X, y, maxiter=1500, tol=1e-3)
```

**Pros:**
- Reliable convergence
- No learning rate tuning
- Fast for smooth objectives
- Default for good reason

**Cons:**
- Memory usage grows with parameters
- Can be slow for very large models

### 2. Muon (State-of-the-art 2024) ⭐

**Best for:** Best performance, limited data

```python
model = DPOSE(layers=(1, 20, 32), optimizer='muon')
model.fit(X, y,
          learning_rate=0.02,    # Higher than Adam!
          beta=0.95,             # Momentum decay
          ns_steps=5,            # Newton-Schulz iterations
          weight_decay=0.0)
```

**Pros:**
- **30%+ sample efficiency** improvement
- State-of-the-art performance
- <3% computational overhead
- Orthogonalizes momentum updates

**Cons:**
- Requires learning rate tuning
- New (2024), less battle-tested

**Performance:**
- CIFAR-10: 23% faster training
- NanoGPT: 1.35x speedup
- Holds current speed records

**How it works:**
1. Accumulates momentum on gradients
2. Orthogonalizes via Newton-Schulz iteration (5 steps)
3. Uses orthogonal version for updates
4. Auto-falls back to Adam for non-2D parameters

**References:**
- Paper: Keller Jordan et al. (2024)
- Blog: https://kellerjordan.github.io/posts/muon/
- Implementation: `optax.contrib.muon`

### 3. L-BFGS

**Best for:** Large networks, memory constraints

```python
model = DPOSE(layers=(1, 20, 32), optimizer='lbfgs')
model.fit(X, y, maxiter=1000, history_size=10)
```

**Pros:**
- Memory-efficient (limited history)
- Often faster than BFGS
- No learning rate tuning

**Cons:**
- Slightly less accurate than BFGS

### 4. Adam

**Best for:** Deep networks, adaptive learning

```python
model = DPOSE(layers=(1, 20, 32), optimizer='adam')
model.fit(X, y,
          learning_rate=1e-3,
          maxiter=2000)
```

**Pros:**
- Adaptive learning rates
- Works well for deep networks
- Well-understood and reliable

**Cons:**
- Requires learning rate tuning
- May need more iterations than BFGS

### 5. SGD

**Best for:** Classical approach, online learning

```python
model = DPOSE(layers=(1, 20, 32), optimizer='sgd')
model.fit(X, y,
          learning_rate=1e-2,
          momentum=0.9,
          maxiter=2000)
```

**Pros:**
- Simple and interpretable
- Works with momentum
- Classical ML choice

**Cons:**
- Requires learning rate tuning
- May need learning rate schedules

### 6. Other Optimizers

**L-BFGS-B** - BFGS with box constraints
```python
model = DPOSE(layers=(1, 20, 32), optimizer='lbfgsb')
```

**Nonlinear CG** - Conjugate gradient
```python
model = DPOSE(layers=(1, 20, 32), optimizer='nonlinear_cg')
```

**Gradient Descent** - Basic baseline
```python
model = DPOSE(layers=(1, 20, 32), optimizer='gradient_descent')
```

## Decision Tree

```
Start
  |
  ├─ Want best performance? ──> Muon (30%+ sample efficiency)
  |
  ├─ Limited training data? ──> Muon
  |
  ├─ Want zero tuning? ──> BFGS (default)
  |
  ├─ Large network? ──> L-BFGS
  |
  ├─ Deep network (>3 layers)? ──> Muon or Adam
  |
  └─ Just starting? ──> BFGS first, then try Muon
```

## Parameter Guide

| Optimizer | Learning Rate | Iterations | Key Parameters |
|-----------|---------------|------------|----------------|
| BFGS | Auto | 1500 | `tol`, `stepsize` |
| **Muon** | **0.02** | **1000** | `beta=0.95`, `ns_steps=5` |
| Adam | 1e-3 | 2000 | `b1`, `b2` |
| SGD | 1e-2 | 2000 | `momentum=0.9` |
| L-BFGS | Auto | 1000 | `history_size=10` |

## Performance Comparison

Benchmark on heteroscedastic regression (200 samples, 500 iterations):

| Optimizer | MAE | Mean σ | Calibration α | Winner |
|-----------|-----|--------|---------------|--------|
| **Muon** | **0.043** | 0.053 | 1.15 | ✓ Best accuracy |
| Adam | 0.053 | 0.065 | 0.83 | Good baseline |
| SGD | 0.053 | 0.067 | 1.02 | Comparable to Adam |
| BFGS | 0.098 | 0.127 | 0.96 | Needs more iterations |

**Key finding:** Muon achieved **18% lower error** than Adam.

## Tuning Tips

### Learning Rates

**BFGS/L-BFGS:**
- No tuning needed (automatic)

**Muon:**
- Start at **0.02** (20x higher than Adam!)
- Try: [0.01, 0.02, 0.03, 0.05]
- Orthogonalization normalizes magnitudes

**Adam:**
- Start at **1e-3**
- Try: [1e-4, 1e-3, 1e-2]
- Adjust by factors of 10

**SGD:**
- Start at **1e-2**
- Try: [1e-3, 1e-2, 1e-1]
- May need learning rate schedule

### Iterations

**Rule of thumb:**
- BFGS/L-BFGS: 1000-1500 iterations
- Muon: 500-1000 iterations (more efficient!)
- Adam/SGD: 1000-2000 iterations

**Check convergence:**
```python
model.fit(X, y)
model.report()  # Check if iterations < maxiter
```

If `iterations == maxiter`, increase `maxiter`.

### Other Parameters

**Muon-specific:**
- `beta=0.95`: Momentum decay (try 0.9-0.95)
- `ns_steps=5`: Orthogonalization accuracy (3-7)
- `weight_decay=0.0`: L2 regularization

**BFGS/L-BFGS:**
- `tol=1e-3`: Convergence tolerance
- `stepsize`: Step size (usually auto)

## Optimizer + Loss Combinations

### Recommended Combinations

| Loss Type | Best Optimizer | Why |
|-----------|---------------|-----|
| CRPS | Muon or BFGS | Robust, works out-of-box |
| NLL (2-stage) | Muon or Adam | Handles pre-training well |
| MSE | BFGS | Simple, no uncertainty |

### CRPS + Muon (Recommended)

```python
model = DPOSE(layers=(1, 20, 32),
              optimizer='muon',
              loss_type='crps')
model.fit(X, y, learning_rate=0.02)
```

**Why:** Best of both worlds - robust loss + efficient optimizer

### NLL + Muon (Advanced)

```python
model = DPOSE(layers=(1, 20, 32),
              optimizer='muon',
              loss_type='nll')
model.fit(X, y,
          pretrain_maxiter=500,    # MSE pre-training
          maxiter=1000,            # NLL fine-tuning
          learning_rate=0.02)
```

**Why:** Two-stage training prevents collapse, Muon accelerates both stages

### MSE + BFGS (Baseline)

```python
model = DPOSE(layers=(1, 20, 32),
              optimizer='bfgs',
              loss_type='mse')
model.fit(X, y)
```

**Why:** No uncertainty training, BFGS guarantees convergence

## Common Issues

### Problem: BFGS is slow

**Solution:** Try L-BFGS or Muon
```python
model = DPOSE(layers=(1, 20, 32), optimizer='lbfgs')
# or
model = DPOSE(layers=(1, 20, 32), optimizer='muon')
```

### Problem: Adam not converging

**Solution:** Tune learning rate or increase iterations
```python
model.fit(X, y, learning_rate=1e-4, maxiter=3000)
```

### Problem: Muon learning rate too high

**Symptoms:** Loss explodes, NaN values

**Solution:** Reduce learning rate
```python
model.fit(X, y, learning_rate=0.01)  # Down from 0.02
```

### Problem: Uncertainty collapse with MSE

**Solution:** Use CRPS or NLL instead
```python
model = DPOSE(layers=(1, 20, 32), loss_type='crps')
```

## Advanced: Custom Optimizer

You can use any `jaxopt` or `optax` optimizer:

```python
import jaxopt
import optax

# In _fit_internal, add your custom optimizer:
elif self.optimizer == 'my_custom':
    solver = jaxopt.OptaxSolver(
        opt=optax.my_custom_optimizer(...),
        fun=objective,
        **kwargs
    )
```

## Examples

See `examples/optimizer_examples.py` for a complete comparison of all optimizers.

## Summary

**Quick recommendations:**

1. **Just starting?** → BFGS (default)
2. **Want best performance?** → Muon
3. **Limited data?** → Muon (30%+ efficiency)
4. **Large network?** → L-BFGS
5. **Deep network?** → Muon or Adam
6. **Need guaranteed convergence?** → BFGS

**Default choice:** BFGS is reliable and works well for most cases.

**Best performance:** Muon (2024) provides state-of-the-art sample efficiency.

---

For more details, see:
- Muon-specific guide: `MUON_OPTIMIZER.md`
- Quick reference: `OPTIMIZER_QUICKSTART.md`
- Examples: `examples/optimizer_examples.py`
