# DPOSE Implementation - Now Fixed! ✅

## What Was Fixed

The original `dpose.py` had a **critical bug** in the NLL loss that made uncertainties meaningless. This has been **completely fixed** to correctly implement the DPOSE method from:

> Kellner, M., & Ceriotti, M. (2024). Uncertainty quantification by direct propagation of shallow ensembles. *Machine Learning: Science and Technology*, 5(3), 035006.

## Quick Start

### Installation Requirements
```bash
pip install jax jaxopt flax scikit-learn matplotlib
```

### Basic Usage
```python
from sklearn.model_selection import train_test_split
from pycse.sklearn.dpose import DPOSE

# Prepare data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Train DPOSE model
model = DPOSE(
    layers=(n_features, 20, 32),  # Last value is ensemble size
    loss_type='nll'                # or 'crps', 'mse'
)
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

# Get predictions with calibrated uncertainties
y_pred, y_std = model.predict(X_test, return_std=True)

# Evaluate
model.print_metrics(X_test, y_test)
```

## Key Features (All Now Working!)

### ✅ Per-Sample Uncertainties
Uncertainties vary across samples based on ensemble spread (heteroscedastic).

### ✅ Multiple Loss Functions
- **NLL**: Standard negative log-likelihood (Eq. 6)
- **CRPS**: More robust alternative (Eq. 18)
- **MSE**: Baseline (no uncertainty training)

### ✅ Automatic Calibration
Post-hoc calibration on validation set ensures uncertainties match empirical errors.

### ✅ Uncertainty Propagation
Propagate uncertainties through non-linear transformations:

```python
# Get ensemble predictions
ensemble = model.predict_ensemble(X)  # (n_samples, n_ensemble)

# Apply transformation
z_ensemble = np.exp(ensemble)  # Or any function f

# Get statistics
z_mean = z_ensemble.mean(axis=1)
z_std = z_ensemble.std(axis=1)
```

### ✅ Comprehensive Diagnostics
```python
metrics = model.uncertainty_metrics(X_test, y_test)
# Returns: RMSE, MAE, NLL, miscalibration area, Z-scores

model.print_metrics(X_test, y_test)  # Pretty-printed report
```

## Verification

Run the test suite to verify everything works:

```bash
cd /Users/jkitchin/Dropbox/python/pycse/src/pycse/sklearn/
python test_dpose_fixed.py
```

Expected output:
- ✅ Heteroscedastic uncertainties detected
- ✅ Ensemble members are diverse
- ✅ Well-calibrated uncertainties
- ✅ Uncertainty propagation works correctly
- 📊 Saves visualization: `dpose_test_results.png`

## What Changed?

See detailed documentation:
- **DPOSE_FIX_SUMMARY.md**: Full list of fixes with code examples
- **DPOSE_BEFORE_AFTER.md**: Visual comparison of broken vs. fixed

### TL;DR - The Critical Fix

**Before (BROKEN):**
```python
sigma = np.std(errors)  # Global constant for all samples ✗
```

**After (FIXED):**
```python
sigma = ensemble_predictions.std(axis=1)  # Per-sample from ensemble ✓
```

This single change:
- Enables heteroscedastic uncertainties
- Maintains ensemble diversity
- Makes uncertainties meaningful
- Follows the paper correctly

## Examples

### Example 1: Heteroscedastic Regression
```python
import jax.numpy as jnp
from pycse.sklearn.dpose import DPOSE

# Generate data with varying noise
x = jnp.linspace(0, 1, 200)[:, None]
noise = 0.01 + 0.1 * x.ravel()  # Increasing noise
y = x.ravel()**2 + noise * jax.random.normal(key, (200,))

# Split data
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

# Train
model = DPOSE(layers=(1, 20, 32), loss_type='nll')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

# Predict
y_pred, y_std = model.predict(x, return_std=True)

# Uncertainty should increase with x!
assert y_std[-1] > y_std[0], "Should capture heteroscedasticity"
```

### Example 2: Uncertainty Propagation
```python
# Trained model from above
ensemble = model.predict_ensemble(x)  # (200, 32)

# Propagate through exp transformation
exp_ensemble = jnp.exp(ensemble)
exp_mean = exp_ensemble.mean(axis=1)
exp_std = exp_ensemble.std(axis=1)

# Plot with uncertainty
plt.fill_between(x.ravel(), exp_mean - 2*exp_std, exp_mean + 2*exp_std, alpha=0.3)
plt.plot(x, exp_mean)
```

### Example 3: CRPS for Robustness
```python
# If NLL degrades accuracy, try CRPS
model_crps = DPOSE(layers=(1, 20, 32), loss_type='crps')
model_crps.fit(X_train, y_train, val_X=X_val, val_y=y_val)

# CRPS is less sensitive to outliers
model_crps.print_metrics(X_test, y_test)
```

## API Reference

### Constructor
```python
DPOSE(
    layers: tuple,           # Network architecture, last value = n_ensemble
    activation=nn.relu,      # Activation function
    seed=19,                 # Random seed
    loss_type='nll',         # 'nll', 'crps', or 'mse'
    min_sigma=1e-6           # Minimum uncertainty for stability
)
```

### Methods

#### fit(X, y, val_X=None, val_y=None, **kwargs)
Train the model with optional calibration.

#### predict(X, return_std=False)
Get predictions (and uncertainties if return_std=True).

#### predict_ensemble(X)
Get full ensemble for uncertainty propagation.

#### uncertainty_metrics(X, y)
Compute UQ metrics (RMSE, MAE, NLL, miscalibration, Z-scores).

#### print_metrics(X, y)
Print formatted diagnostic report.

#### plot(X, y, distribution=False, ax=None)
Visualize predictions with uncertainty bands.

## Troubleshooting

### Issue: Uncertainties are uniform
**Symptom:** All uncertainties nearly identical
**Cause:** Ensemble collapsed
**Fix:**
- Increase ensemble size (try 32 or 64)
- Use NLL or CRPS loss (not MSE)
- Check that training converged

### Issue: Underconfident predictions
**Symptom:** Z-score std > 1.5, uncertainties too large
**Fix:**
- Provide validation set for calibration
- Try CRPS loss instead of NLL
- Increase training iterations

### Issue: Overconfident predictions
**Symptom:** Z-score std < 0.7, uncertainties too small
**Fix:**
- Provide validation set for calibration
- Increase ensemble size
- Check for overfitting

## Performance Tips

### Ensemble Size
- **Small (8-16)**: Fast but may underestimate uncertainty
- **Medium (32)**: Good balance (recommended)
- **Large (64-128)**: Best uncertainty, minimal extra cost (weights shared!)

### Loss Function
- **Start with NLL**: Usually works well
- **If RMSE degrades**: Switch to CRPS
- **For baseline**: Use MSE (but no uncertainty training)

### Calibration
- **Always provide validation set** for post-hoc calibration
- ~20% of training data usually sufficient
- Check calibration with `print_metrics()`

## Citation

If you use this code, please cite the original DPOSE paper:

```bibtex
@article{kellner2024uncertainty,
  title={Uncertainty quantification by direct propagation of shallow ensembles},
  author={Kellner, Matthias and Ceriotti, Michele},
  journal={Machine Learning: Science and Technology},
  volume={5},
  number={3},
  pages={035006},
  year={2024},
  publisher={IOP Publishing}
}
```

## Acknowledgments

Original DPOSE method by Kellner & Ceriotti (2024).
Implementation fixed to correctly follow their paper.

---

**Questions?** Check the test file `test_dpose_fixed.py` for more examples.

**Found a bug?** The implementation now correctly follows Kellner & Ceriotti (2024). Compare with their paper if in doubt.
