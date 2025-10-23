# DPOSE: Direct Propagation of Shallow Ensembles

JAX/Flax implementation of DPOSE for uncertainty quantification in regression tasks.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Key Features](#key-features)
- [Optimizers](#optimizers)
- [Loss Functions](#loss-functions)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)
- [Documentation](#documentation)
- [Reference](#reference)

## Quick Start

```python
import jax
import numpy as np
from pycse.sklearn.dpose import DPOSE
from sklearn.model_selection import train_test_split

# Generate example data
key = jax.random.PRNGKey(42)
X = np.linspace(0, 1, 200)[:, None]
noise = 0.01 + 0.08 * X.ravel()
y = X.ravel()**(1/3) + noise * jax.random.normal(key, (200,))

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Train model
model = DPOSE(layers=(1, 20, 32), loss_type='crps')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

# Get predictions with uncertainty
y_pred, y_std = model.predict(X_val, return_std=True)

# Visualize
model.plot(X, y, distribution=True)
```

## Installation

```bash
pip install jax jaxopt optax flax scikit-learn matplotlib
```

## Key Features

- **Shallow Ensemble**: Only last layer differs across members (efficient!)
- **Multiple Loss Functions**: CRPS (recommended), NLL with auto pre-training, MSE
- **Automatic Calibration**: Post-hoc uncertainty calibration on validation data
- **8 Optimizers**: BFGS, Muon, Adam, L-BFGS, SGD, and more
- **Uncertainty Propagation**: Ensemble predictions for derived quantities

## Optimizers

### Quick Optimizer Guide

| Optimizer | Use When | Learning Rate | Performance |
|-----------|----------|---------------|-------------|
| **`muon`** | Best performance | 0.02 | ⭐⭐⭐⭐⭐ 18% better than Adam |
| **`bfgs`** (default) | Reliable, no tuning | Auto | ⭐⭐⭐⭐ |
| `lbfgs` | Large networks | Auto | ⭐⭐⭐⭐ |
| `adam` | Deep networks | 1e-3 | ⭐⭐⭐ |
| `sgd` | Classical approach | 1e-2 | ⭐⭐⭐ |

### Muon Optimizer (State-of-the-art 2024)

**Best for:** Limited data, best performance (30%+ sample efficiency)

```python
model = DPOSE(layers=(1, 20, 32), optimizer='muon')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val,
          learning_rate=0.02,    # Higher than Adam!
          beta=0.95,             # Momentum decay
          ns_steps=5)            # Newton-Schulz iterations
```

**Performance:**
- 30%+ sample efficiency improvement
- <3% computational overhead
- Won MLCommons 2024 AlgoPerf Challenge (via Schedule-Free variant)
- Current speed records for CIFAR-10 and NanoGPT

**How it works:** Orthogonalizes momentum updates using Newton-Schulz iteration for better gradient geometry.

### Other Optimizers

```python
# BFGS (default, reliable)
model = DPOSE(layers=(1, 20, 32))
model.fit(X, y)

# Adam (adaptive learning)
model = DPOSE(layers=(1, 20, 32), optimizer='adam')
model.fit(X, y, learning_rate=1e-3)

# L-BFGS (memory-efficient)
model = DPOSE(layers=(1, 20, 32), optimizer='lbfgs')
model.fit(X, y)
```

## Loss Functions

### CRPS (Recommended)

**Continuous Ranked Probability Score** - Most robust

```python
model = DPOSE(layers=(1, 20, 32), loss_type='crps')
model.fit(X, y)
```

✅ Works out-of-the-box
✅ Prevents uncertainty collapse
✅ Less sensitive to outliers

### NLL (Advanced)

**Negative Log-Likelihood** - Automatic two-stage training

```python
model = DPOSE(layers=(1, 20, 32), loss_type='nll')
model.fit(X, y, val_X=X_val, val_y=y_val,
          pretrain_maxiter=500,    # MSE pre-training (automatic)
          maxiter=1000)            # NLL fine-tuning
```

Stage 1: MSE pre-training → good predictions
Stage 2: NLL fine-tuning → calibrated uncertainties

### MSE (Baseline)

**Mean Squared Error** - Standard regression (no uncertainty)

```python
model = DPOSE(layers=(1, 20, 32), loss_type='mse')
model.fit(X, y)
```

## Examples

All examples are in the `examples/` directory:

```bash
# Compare all optimizers
python examples/optimizer_examples.py

# Test Muon optimizer
python examples/test_muon.py

# Quick smoke test
python examples/test_optimizers.py

# Complete demonstration
python examples/demo_dpose_datasets.py
```

See `examples/README.md` for details.

## Advanced Usage

### Uncertainty Propagation

For derived quantities `z = f(y)`:

```python
# Get full ensemble predictions
ensemble = model.predict_ensemble(X)  # (n_samples, n_ensemble)

# Apply function to each ensemble member
z_ensemble = f(ensemble)

# Get propagated uncertainty
z_mean = z_ensemble.mean(axis=1)
z_std = z_ensemble.std(axis=1)
```

### Model Diagnostics

```python
# Optimization report
model.report()
# Shows: iterations, final loss, optimizer, calibration

# Uncertainty metrics
model.print_metrics(X_val, y_val)
# Shows: RMSE, MAE, NLL, z-scores, miscalibration area
```

### Custom Training

```python
model = DPOSE(
    layers=(input_dim, hidden_size, ensemble_size),
    activation=nn.relu,           # Default activation
    loss_type='crps',             # 'crps', 'nll', or 'mse'
    optimizer='muon',             # See optimizers above
    seed=19,                      # Random seed
    min_sigma=1e-3               # Numerical stability
)

model.fit(
    X_train, y_train,
    val_X=X_val, val_y=y_val,    # For calibration (recommended)
    maxiter=1500,                 # Max iterations
    learning_rate=0.02,           # Optimizer-specific
    **optimizer_kwargs            # Optimizer-specific params
)
```

## Documentation

### In This Directory

- **`README.md`** (you are here) - Main guide

### In `docs/` Directory

**Optimizer Guides:**
- `docs/OPTIMIZERS.md` - Complete optimizer reference
- `docs/MUON_OPTIMIZER.md` - Muon optimizer deep dive
- `docs/OPTIMIZER_QUICKSTART.md` - Quick decision guide

**Troubleshooting:**
- `docs/NLL_AUTO_PRETRAIN.md` - NLL two-stage training
- `docs/WHY_NLL_FAILS.md` - NLL common issues
- `docs/ENSEMBLE_COLLAPSE_TROUBLESHOOTING.md` - Fix collapsed ensembles

**Historical:**
- `docs/SUMMARY_OF_FIXES.md` - Development history
- `docs/README.md` - Full docs index

### In `examples/` Directory

- `examples/README.md` - Examples guide
- `examples/*.py` - Working example scripts

## Architecture

DPOSE uses a **shallow ensemble** approach:

```
Input (n_features)
    ↓
Hidden Layers (shared across all ensemble members)
    ↓
Output Layer (n_ensemble separate outputs)
```

Only the final layer differs, making it **much more efficient** than training N separate networks.

## Performance Benchmark

Heteroscedastic regression (200 samples, 500 iterations):

| Optimizer | MAE | Improvement vs Adam |
|-----------|-----|---------------------|
| **Muon** | **0.043** | **18% better** |
| Adam | 0.053 | baseline |
| BFGS | 0.098 | needs more iterations |

## Common Patterns

### Pattern 1: Default (Reliable)

```python
model = DPOSE(layers=(1, 20, 32))
model.fit(X, y, val_X=X_val, val_y=y_val)
```

### Pattern 2: Best Performance

```python
model = DPOSE(layers=(1, 20, 32), optimizer='muon', loss_type='crps')
model.fit(X, y, val_X=X_val, val_y=y_val, learning_rate=0.02)
```

### Pattern 3: NLL with Muon

```python
model = DPOSE(layers=(1, 20, 32), optimizer='muon', loss_type='nll')
model.fit(X, y, val_X=X_val, val_y=y_val,
          pretrain_maxiter=500, maxiter=1000, learning_rate=0.02)
```

## Troubleshooting

### Problem: Uncertainties are zero

**Solution:** Use CRPS loss or increase ensemble size

```python
model = DPOSE(layers=(1, 20, 64), loss_type='crps')  # Larger ensemble
```

See: `docs/ENSEMBLE_COLLAPSE_TROUBLESHOOTING.md`

### Problem: NLL gives NaN

**Solution:** Already automatic! NLL auto pre-trains with MSE

```python
model = DPOSE(layers=(1, 20, 32), loss_type='nll')
model.fit(X, y)  # Automatically does two-stage training
```

See: `docs/WHY_NLL_FAILS.md`

### Problem: Poor calibration

**Solution:** Provide validation data

```python
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
# Automatically calibrates uncertainties using α factor
```

## Reference

### DPOSE Paper

Kellner, M., & Ceriotti, M. (2024). Uncertainty quantification by direct propagation of shallow ensembles. *Machine Learning: Science and Technology*, 5(3), 035006.

### Muon Optimizer

Keller Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks.
- Blog: https://kellerjordan.github.io/posts/muon/
- GitHub: https://github.com/KellerJordan/Muon
- Implementation: `optax.contrib.muon`

### Citation

```bibtex
@article{kellner2024uncertainty,
  title={Uncertainty quantification by direct propagation of shallow ensembles},
  author={Kellner, M. and Ceriotti, M.},
  journal={Machine Learning: Science and Technology},
  volume={5},
  number={3},
  pages={035006},
  year={2024}
}
```

## Quick Reference Card

| Task | Command |
|------|---------|
| Basic training | `model = DPOSE(layers=(1,20,32)); model.fit(X, y)` |
| Best performance | `DPOSE(layers=(1,20,32), optimizer='muon')` |
| With validation | `model.fit(X_train, y_train, val_X=X_val, val_y=y_val)` |
| Get uncertainty | `y_pred, y_std = model.predict(X, return_std=True)` |
| Propagate uncertainty | `ensemble = model.predict_ensemble(X); z = f(ensemble)` |
| Diagnostics | `model.report(); model.print_metrics(X, y)` |
| Visualize | `model.plot(X, y, distribution=True)` |

## Support

- **Examples:** See `examples/`
- **Detailed guides:** See `docs/`
- **Issues:** Check `docs/README.md` for common problems

---

**Version:** Enhanced with Muon optimizer (2024)
**License:** Part of pycse package
**Dependencies:** JAX, Optax, Flax, scikit-learn
