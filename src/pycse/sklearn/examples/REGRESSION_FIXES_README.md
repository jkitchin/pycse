# Linear Regression Bug Fixes - October 2025

This directory contains documentation and tests for critical bug fixes applied to the linear regression functions in PYCSE.py.

## Fixed Functions

1. **regress()** - Linear least squares regression with confidence intervals
2. **predict()** - Prediction intervals for linear regression
3. **nlpredict()** - Prediction intervals for nonlinear regression

## Files

### Documentation
- **FIXES_APPLIED.md** - Complete documentation of all bugs found and fixes applied
  - Detailed explanation of each issue
  - Before/after code comparisons
  - Impact assessment
  - Verification results

### Tests
- **test_fixes.py** - Comprehensive test suite verifying all fixes
  - Tests regress() DOF correction
  - Tests predict() variance, covariance, and formula fixes
  - Manual calculation verification
  - Empirical coverage check (~95% as expected)
  - Run with: `python3 test_fixes.py`

### Visualization
- **before_after_comparison.py** - Script to generate visual comparison
  - Shows old (buggy) vs new (fixed) prediction intervals
  - Demonstrates the severity of the bugs
  - Creates publication-quality figure

- **before_after_comparison.png** - Visual comparison figure
  - Shows old intervals were 91% too narrow
  - Effect most pronounced near the mean (10x difference)

## Quick Summary

### What Was Fixed

| Function | Issue | Impact |
|----------|-------|--------|
| regress() | DOF: used n-k-1 instead of n-k | Minor (0.05% too wide) |
| predict() | Biased variance: sse/n instead of sse/(n-k) | 4% underestimation |
| predict() | Factor of 2 error in covariance | 50% underestimation |
| predict() | Wrong prediction formula | Severe (10x at mean) |
| nlpredict() | Loss convention ambiguity | Convention mismatch |
| nlpredict() | Biased variance estimator | 4% underestimation |

### Key Findings

- **Old predict() intervals were 91% too narrow** - critically dangerous!
- At sample mean: intervals were **10x too narrow**
- All functions now match scipy.stats behavior
- Empirical coverage now matches nominal 95% level

## Verification

All fixes have been extensively tested:
```bash
cd /Users/jkitchin/Dropbox/python/pycse/src/pycse/sklearn/examples
python3 test_fixes.py
```

Expected output:
- ✓ regress() matches scipy.stats.linregress
- ✓ predict() manual calculations match code
- ✓ Parameter SE consistent between regress() and predict()
- ✓ Empirical coverage ~95%

## References

- See FIXES_APPLIED.md for complete technical details
- See before_after_comparison.png for visual impact
- Modified file: /Users/jkitchin/Dropbox/python/pycse/src/pycse/PYCSE.py (lines 167-170, 210-260, 301-364)

---

**Date:** October 23, 2025
**Status:** All fixes verified and applied ✓
