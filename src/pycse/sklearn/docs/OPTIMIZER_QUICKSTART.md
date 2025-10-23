# DPOSE Optimizer Quick Reference

## Basic Usage

```python
from pycse.sklearn.dpose import DPOSE

# Default (BFGS - recommended)
model = DPOSE(layers=(1, 20, 32))
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
```

## All Available Optimizers

```python
# 1. BFGS (default, best for most cases)
model = DPOSE(layers=(1, 20, 32), optimizer='bfgs')
model.fit(X, y, maxiter=1500, tol=1e-3)

# 2. L-BFGS (memory-efficient)
model = DPOSE(layers=(1, 20, 32), optimizer='lbfgs')
model.fit(X, y, maxiter=1000)

# 3. L-BFGS-B (with box constraints)
model = DPOSE(layers=(1, 20, 32), optimizer='lbfgsb')
model.fit(X, y, maxiter=1000)

# 4. Nonlinear CG
model = DPOSE(layers=(1, 20, 32), optimizer='nonlinear_cg')
model.fit(X, y, maxiter=1000)

# 5. Adam (adaptive learning rate)
model = DPOSE(layers=(1, 20, 32), optimizer='adam')
model.fit(X, y, maxiter=2000, learning_rate=1e-3)

# 6. SGD with momentum
model = DPOSE(layers=(1, 20, 32), optimizer='sgd')
model.fit(X, y, maxiter=2000, learning_rate=1e-2, momentum=0.9)

# 7. Muon (state-of-the-art 2024)
model = DPOSE(layers=(1, 20, 32), optimizer='muon')
model.fit(X, y, maxiter=1000, learning_rate=0.02, beta=0.95)

# 8. Gradient Descent (basic)
model = DPOSE(layers=(1, 20, 32), optimizer='gradient_descent')
model.fit(X, y, maxiter=1000)
```

## Decision Tree

```
Start
  |
  ├─ Want state-of-the-art? ──> Muon (best sample efficiency)
  |
  ├─ Small/medium network? ──> BFGS (default, reliable)
  |
  ├─ Large network? ──> L-BFGS
  |
  ├─ Deep network (>3 layers)? ──> Muon or Adam
  |
  ├─ Limited training data? ──> Muon (30%+ sample efficiency)
  |
  └─ Just experimenting? ──> Try BFGS first, then Muon
```

## Common Parameters

| Parameter | BFGS/L-BFGS | Adam/SGD | Muon | Default |
|-----------|-------------|----------|------|---------|
| `maxiter` | ✓ | ✓ | ✓ | 1500 |
| `tol` | ✓ | ✓ | ✓ | 1e-3 |
| `learning_rate` | ✗ | ✓ | ✓ | 1e-3 (Adam), 1e-2 (SGD), 0.02 (Muon) |
| `momentum` / `beta` | ✗ | ✓ (SGD) | ✓ | 0.9 (SGD), 0.95 (Muon) |
| `ns_steps` | ✗ | ✗ | ✓ | 5 |
| `weight_decay` | ✗ | ✗ | ✓ | 0.0 |

## Tips

1. **Start with defaults**: BFGS with default parameters works well
2. **Try Muon for best results**: Especially with limited data (30%+ sample efficiency)
3. **Switch to L-BFGS**: If memory is an issue
4. **Try Adam**: If BFGS gets stuck or network is deep
5. **Tune learning rates**:
   - Adam: Start at 1e-3, adjust by 10x
   - SGD: Start at 1e-2, adjust by 10x
   - Muon: Start at 0.02 (20x higher than Adam!)
6. **Increase maxiter**: If optimization hasn't converged (check with `model.report()`)

## Checking Convergence

```python
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
model.report()  # Shows iterations, final loss, optimizer used
```

Look for:
- **Iterations < maxiter**: Converged successfully ✓
- **Iterations = maxiter**: May need more iterations ⚠
- **Calibration α ≈ 1.0**: Well-calibrated uncertainties ✓
