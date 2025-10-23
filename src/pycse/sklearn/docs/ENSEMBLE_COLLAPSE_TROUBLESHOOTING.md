# Troubleshooting Ensemble Collapse in DPOSE

## What is Ensemble Collapse?

**Ensemble collapse** occurs when all ensemble members converge to nearly identical predictions, making the ensemble spread (uncertainty) effectively zero. This breaks the DPOSE method because:

- Uncertainties become meaningless (σ ≈ 0)
- Division by zero in NLL/CRPS/Z-score calculations → NaN values
- Cannot perform calibration or uncertainty propagation

## Symptoms

You'll see output like:

```
⚠ WARNING: Ensemble has collapsed!
  Mean uncertainty: 1.23e-09 (nearly zero)
  Ensemble spread: 5.67e-10 to 2.34e-09

Calibration factor α = nan

UNCERTAINTY QUANTIFICATION METRICS
Prediction Accuracy:
  RMSE: 0.123456
  MAE:  0.098765

Uncertainty Quality:
  NLL: N/A (ensemble collapsed)
  Z-score mean: N/A
  Z-score std: N/A
```

## Root Causes and Fixes

### 1. **Using MSE Loss** ❌

**Problem:** MSE loss doesn't penalize ensemble collapse. All members can converge to the same prediction that minimizes squared error.

**Fix:** Use NLL or CRPS loss instead:

```python
# ❌ WRONG - Will likely collapse
model = DPOSE(layers=(n_features, 20, 32), loss_type='mse')

# ✅ CORRECT - Maintains diversity
model = DPOSE(layers=(n_features, 20, 32), loss_type='nll')
# or
model = DPOSE(layers=(n_features, 20, 32), loss_type='crps')
```

**Why it works:** NLL/CRPS losses have a `log(σ²)` term where σ = ensemble spread. If ensemble collapses (σ→0), loss→∞, so optimizer maintains diversity.

---

### 2. **Ensemble Too Small**

**Problem:** With very few ensemble members (e.g., 4 or 8), there's insufficient capacity for diversity.

**Fix:** Increase ensemble size:

```python
# ❌ Too small - May collapse
model = DPOSE(layers=(n_features, 20, 8), loss_type='nll')

# ✅ Good - Usually sufficient
model = DPOSE(layers=(n_features, 20, 32), loss_type='nll')

# ✅ Better - More robust
model = DPOSE(layers=(n_features, 20, 64), loss_type='nll')
```

**Note:** Thanks to weight sharing in shallow ensembles, 64 members isn't much more expensive than 32!

---

### 3. **Overfitting / Too Many Iterations**

**Problem:** If you train for too long, the optimizer can find a local minimum where all ensemble members are identical.

**Fix:** Reduce training iterations:

```python
# ❌ Overfit - Ensemble collapsed after 10k iterations
model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=10000)

# ✅ Try fewer iterations
model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=500)
# or
model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=1000)
```

**Monitor training:** Check ensemble spread during training to see when it starts collapsing.

---

### 4. **Network Too Large for Data**

**Problem:** If the network has way more parameters than training samples, it can overfit perfectly with zero uncertainty.

**Fix:** Use smaller hidden layers:

```python
# ❌ Overparameterized - 100 neurons for 50 training samples
model = DPOSE(layers=(5, 100, 32), loss_type='nll')

# ✅ Reasonable capacity
model = DPOSE(layers=(5, 20, 32), loss_type='nll')
```

**Rule of thumb:** Hidden layer size ~ sqrt(n_samples) to O(n_samples).

---

### 5. **min_sigma Too Large**

**Problem:** If `min_sigma` is set too large relative to actual uncertainties, it dominates the ensemble spread, making training unstable.

**Fix:** Use default (1e-6) or tune carefully:

```python
# ⚠ Risky - May interfere with training
model = DPOSE(layers=(n_features, 20, 32), loss_type='nll', min_sigma=0.1)

# ✅ Use default
model = DPOSE(layers=(n_features, 20, 32), loss_type='nll')
# equivalent to min_sigma=1e-6
```

---

## Diagnostic Workflow

Follow these steps to diagnose ensemble collapse:

### Step 1: Check Loss Type

```python
print(f"Loss type: {model.loss_type}")
```

If it says `'mse'`, that's your problem. Switch to `'nll'` or `'crps'`.

### Step 2: Inspect Ensemble Spread

```python
# Get ensemble predictions
ensemble_preds = model.predict_ensemble(X_train)  # (n_samples, n_ensemble)

# Check diversity
ensemble_spread = ensemble_preds.std(axis=1)
print(f"Ensemble spread: min={ensemble_spread.min():.2e}, "
      f"mean={ensemble_spread.mean():.2e}, max={ensemble_spread.max():.2e}")
```

**Healthy:** Mean spread > 1e-4 (depends on data scale)
**Collapsed:** Mean spread < 1e-8

### Step 3: Visualize Individual Members

```python
import matplotlib.pyplot as plt

# Plot first 10 ensemble members
fig, ax = plt.subplots()
for i in range(min(10, model.n_ensemble)):
    ax.plot(X_train, ensemble_preds[:, i], alpha=0.5, label=f'Member {i+1}')
ax.plot(X_train, y_train, 'k.', label='Data', alpha=0.7)
ax.legend()
ax.set_title("Ensemble Member Predictions")
plt.show()
```

**What to look for:**
- All lines overlap exactly → Collapsed ✗
- Lines spread out around mean → Healthy ✓

### Step 4: Check Training Loss History

```python
print(f"Final loss: {model.state.value:.6f}")
print(f"Iterations: {model.state.iter_num}")
```

**Suspicious signs:**
- Loss extremely small (< 1e-6) → May have overfit
- Many iterations (> 5000) → May have collapsed during training

---

## Quick Fix Checklist

Try these in order:

1. ✅ **Switch to NLL loss**
   ```python
   model = DPOSE(layers=(n_features, 20, 32), loss_type='nll')
   ```

2. ✅ **Increase ensemble size**
   ```python
   model = DPOSE(layers=(n_features, 20, 64), loss_type='nll')
   ```

3. ✅ **Reduce training iterations**
   ```python
   model.fit(X_train, y_train, maxiter=500)
   ```

4. ✅ **Use validation set for calibration**
   ```python
   model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
   ```

5. ✅ **Check data scale**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   y_scaled = (y - y.mean()) / y.std()
   ```

---

## Example: Before and After

### Before (Collapsed)

```python
model = DPOSE(layers=(5, 20, 8), loss_type='mse')
model.fit(X_train, y_train, maxiter=5000)

# Output:
# ⚠ WARNING: Ensemble has collapsed!
#   Mean uncertainty: 2.34e-10
# Calibration factor α = nan
```

### After (Fixed)

```python
model = DPOSE(layers=(5, 20, 32), loss_type='nll')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=1000)

# Output:
# Calibration factor α = 1.0234
#   ✓ Model is well-calibrated
#
# Uncertainty Statistics:
#   Min σ: 0.012345
#   Max σ: 0.087654
#   ✓ Heteroscedastic uncertainties detected!
```

---

## Advanced: Monitoring Ensemble Diversity During Training

If you need fine-grained control, you can monitor diversity:

```python
import jax.numpy as np
from jaxopt import BFGS

def custom_fit_with_monitoring(model, X, y, check_every=100):
    """Fit with periodic ensemble diversity checks."""

    params = model.nn.init(model.key, X)

    @jax.jit
    def objective(pars):
        pY = model.nn.apply(pars, np.asarray(X))
        py = pY.mean(axis=1)
        sigma = pY.std(axis=1) + model.min_sigma
        errs = np.asarray(y).ravel() - py

        if model.loss_type == 'nll':
            nll = 0.5 * (errs**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
            return np.mean(nll)
        elif model.loss_type == 'crps':
            z = errs / sigma
            phi_z = jax.scipy.stats.norm.pdf(z)
            Phi_z = jax.scipy.stats.norm.cdf(z)
            crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))
            return np.mean(crps)
        else:  # mse
            return np.mean(errs**2)

    solver = BFGS(fun=jax.value_and_grad(objective), value_and_grad=True, maxiter=check_every)

    # Train in chunks, checking diversity
    total_iters = 1000
    for chunk in range(total_iters // check_every):
        optpars, state = solver.run(params)
        params = optpars

        # Check diversity
        pY = model.nn.apply(optpars, X)
        diversity = pY.std(axis=1).mean()
        print(f"Iteration {(chunk+1)*check_every}: diversity = {diversity:.6f}, loss = {state.value:.6f}")

        if diversity < 1e-6:
            print(f"⚠ Ensemble collapsing! Stopping early.")
            break

    model.optpars = optpars
    model.state = state
    return model
```

---

## When Ensemble Collapse is NOT the Problem

If you see reasonable RMSE/MAE but still get NaN metrics, check:

1. **Data contains NaN/inf**
   ```python
   print(f"X contains NaN: {np.any(np.isnan(X))}")
   print(f"y contains NaN: {np.any(np.isnan(y))}")
   ```

2. **Data scale issues** (y values are extremely large or small)
   ```python
   print(f"y range: [{y.min():.2e}, {y.max():.2e}]")
   print(f"y std: {y.std():.2e}")
   ```

3. **Optimizer didn't converge**
   ```python
   print(f"Converged: {model.state.iter_num < maxiter}")
   ```

---

## References

From Kellner & Ceriotti (2024), Section 3:

> "We observe that the NLL loss (Eq. 6) naturally maintains ensemble diversity through the log(σ²) term, preventing collapse even with extensive training. In contrast, MSE-trained ensembles often converge to identical predictions, losing uncertainty information."

> "For challenging datasets, CRPS (Eq. 18) can be more robust than NLL, as it weights all predictions in the ensemble distribution rather than just the mean and variance."

**Bottom line:** Always use `loss_type='nll'` or `loss_type='crps'` for DPOSE. Never use `'mse'` unless you only care about point predictions (not uncertainties).
