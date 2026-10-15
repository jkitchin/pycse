# Regression and prediction intervals

All functions are importable from `pycse`. `alpha=0.05` gives 95% intervals; the t
multiplier uses `n - p` degrees of freedom.

## Contents
- Linear: `regress`, `predict`
- Polynomial: `polyfit`, `polyval`
- Nonlinear: `nlinfit`, `nlpredict`
- Which standard error each function returns
- Model selection: `bic`, `lbic`, `Rsquared`

## Linear: `regress(A, y, alpha=0.05)`

Solves `A @ b = y` by least squares. `A` holds one column per basis function.

```python
import numpy as np
from pycse import regress, predict

A = np.column_stack([x**0, x, x**2])
b, bint, se = regress(A, y)
# b: (p,) parameters; bint: (p, 2) intervals, row i = [lower, upper] for b[i]; se: (p,)
# regress(A, y, alpha=None) skips the statistics and returns (b, None, None)

Anew = np.column_stack([xnew**0, xnew, xnew**2])
yp, yint, pred_se = predict(A, y, b, Anew)
# yint: (n, 2) prediction interval; with k outputs (y of shape (N, k)) it is (n, k, 2)
```

## Polynomial: `polyfit(x, y, deg)` and `polyval(p, x, X, y)`

`polyfit` builds a Vandermonde matrix (highest power first, like `numpy.polyfit`) and calls
`regress`. `polyval` needs the original data `X, y` to compute the intervals.

```python
from pycse import polyfit, polyval

p, pint, se = polyfit(x, y, 2)
yp, yint, pred_se = polyval(p, xnew, x, y)  # note argument order: p, new x, original x, y
```

## Nonlinear: `nlinfit(model, x, y, p0)` and `nlpredict(X, y, model, popt, xnew)`

`model(x, *params)` has the same form as for `scipy.optimize.curve_fit`; extra keyword
arguments to `nlinfit` go to `curve_fit` (e.g. `bounds=`, `sigma=`).

```python
from pycse import nlinfit, nlpredict

p, pint, se = nlinfit(model, x, y, p0)
yp, yint, se_mean = nlpredict(x, y, model, p, xnew)
```

`nlpredict` assumes the parameters minimized ½·SSE (what `curve_fit` does). If they came from a
different objective, pass it as `loss=lambda *p: ...`; it must be proportional to the sum of
squared residuals.

## Which standard error is returned

| Function | Third return value |
|---|---|
| `predict`, `polyval` | prediction standard error for a new observation, `sqrt(sigma**2 + se_mean**2)` |
| `nlpredict` | standard error of the fitted mean only; the interval uses `sqrt(se**2 + sigma**2)` |

Use `yint` when you need the interval; do not rebuild it from the third return value.

## Interval shape changed in 2.12

`predict`, `polyval` and `nlpredict` return `yint` with shape `(n, 2)`. Before 2.12,
`predict` and `polyval` returned `(2, n)`. Old code like `lower, upper = yint` now unpacks
rows instead of bounds; use `lower, upper = yint[:, 0], yint[:, 1]` (or `yint.T`).

## Model selection

- `bic(x, y, model, popt)`: Bayesian information criterion for a nonlinear model.
- `lbic(A, y, b)`: the same for a linear model.
- `Rsquared(y, y_pred)`: coefficient of determination.

Lower BIC is better. Compare models fitted to the same data.
