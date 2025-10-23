# DPOSE Documentation Archive

This directory contains historical and diagnostic documentation related to DPOSE development and troubleshooting.

## Contents

### NLL Training Documentation

- **`NLL_AUTO_PRETRAIN.md`** - Two-stage NLL training (MSE → NLL)
- **`WHY_NLL_FAILS.md`** - Common NLL pitfalls and solutions
- **`WHEN_DOES_NLL_WORK.md`** - Best practices for NLL loss

### Ensemble Collapse Diagnostics

- **`ENSEMBLE_COLLAPSE_TROUBLESHOOTING.md`** - Diagnosing and fixing ensemble collapse

### Development History

- **`DPOSE_BEFORE_AFTER.md`** - Comparison of implementations
- **`DPOSE_FIX_SUMMARY.md`** - Summary of bug fixes
- **`README_DPOSE_FIX.md`** - Detailed fix documentation
- **`SUMMARY_OF_FIXES.md`** - Comprehensive fix summary

## When to Use These Docs

### You should read NLL docs if:
- ❌ NLL loss is producing NaN or inf values
- ❌ Training fails with NLL but works with CRPS
- ❌ Uncertainties collapse to zero with NLL
- ✅ You want to understand two-stage training

### You should read ensemble collapse docs if:
- ❌ Model reports "Ensemble has collapsed"
- ❌ All predictions have zero uncertainty
- ❌ Ensemble members are identical
- ✅ You want to understand what causes collapse

### You should read development history if:
- ✅ You're curious about how issues were fixed
- ✅ You want to understand design decisions
- ✅ You're debugging similar issues in your own code

## Quick Solutions

### Problem: NLL gives NaN

**Solution:** Use two-stage training (automatic with `loss_type='nll'`)

```python
model = DPOSE(layers=(1, 20, 32), loss_type='nll')
model.fit(X, y, val_X=X_val, val_y=y_val)  # Auto pre-trains with MSE
```

See: `NLL_AUTO_PRETRAIN.md`, `WHY_NLL_FAILS.md`

### Problem: Ensemble collapse

**Solution:** Use CRPS loss or increase ensemble size

```python
# Option 1: Use CRPS (recommended)
model = DPOSE(layers=(1, 20, 32), loss_type='crps')

# Option 2: Larger ensemble
model = DPOSE(layers=(1, 20, 64), loss_type='nll')  # 64 instead of 32
```

See: `ENSEMBLE_COLLAPSE_TROUBLESHOOTING.md`

### Problem: Poor uncertainty calibration

**Solution:** Provide validation data for post-hoc calibration

```python
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
# Automatically applies calibration factor α
```

## Recommended Reading Order

1. **Start here:** `NLL_AUTO_PRETRAIN.md` - Understand the two-stage approach
2. **If using NLL:** `WHY_NLL_FAILS.md` - Avoid common pitfalls
3. **If encountering issues:** `ENSEMBLE_COLLAPSE_TROUBLESHOOTING.md`
4. **For historical context:** `SUMMARY_OF_FIXES.md`

## Not Sure Where to Look?

**Use the main docs instead:**
- Parent directory: `../README.md` - Main guide
- `../OPTIMIZERS.md` - Optimizer selection
- `../MUON_OPTIMIZER.md` - Muon details
- `../OPTIMIZER_QUICKSTART.md` - Quick reference

These historical docs are primarily for:
- Debugging specific issues
- Understanding design rationale
- Learning from past problems

For everyday use, stick to the main documentation in the parent directory!
