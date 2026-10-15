---
name: pycse
description: Writes Python code with the pycse library (Python Computations in Science and Engineering). Covers linear, polynomial and nonlinear regression with parameter confidence intervals and prediction intervals (regress, polyfit, nlinfit, predict, nlpredict), ODE initial value problems (ivp), design of experiments (Latin squares, surface response), active learning, and sklearn-compatible models with uncertainty (DPOSE, KAN, LLPR, monotonic/periodic/convex neural networks, SISSO, ZENN). Use when the user imports pycse, asks for fits with confidence or prediction intervals, wants to design experiments, or needs a regression model that reports uncertainty.
---

# pycse

pycse wraps numpy/scipy/sklearn/JAX for engineering calculations, with an emphasis on
uncertainty: fits return confidence intervals and models return a standard deviation.
Install with `pip install pycse`; it needs Python 3.12 or newer.

## Regression with confidence intervals

```python
import numpy as np
from pycse import nlinfit, nlpredict


def model(x, a, k):
    return a * np.exp(-k * x)


x = np.linspace(0, 5, 20)
y = model(x, 2.0, 0.7) + 0.02 * np.random.default_rng(0).standard_normal(20)

p, pint, se = nlinfit(model, x, y, [1.0, 1.0])  # argument order: model, x, y, p0
# p: fitted parameters; pint: (n_params, 2) 95% intervals; se: standard errors

xnew = np.linspace(0, 6, 50)
yp, yint, se_mean = nlpredict(x, y, model, p, xnew)
# yint has shape (len(xnew), 2): yint[:, 0] lower, yint[:, 1] upper
```

The linear counterparts are `regress(A, y)` (design matrix `A`) with `predict(A, y, b, Anew)`,
and `polyfit(x, y, deg)` with `polyval(p, xnew, x, y)`. All take `alpha=0.05` for 95% intervals.

**Interval shape:** since pycse 2.12, the prediction intervals from `predict`, `polyval` and
`nlpredict` all have shape `(n, 2)`. Code written for older versions used `(2, n)`
(`lower, upper = yint`); write `yint[:, 0]`, `yint[:, 1]` instead.

Signatures, model selection (`bic`, `lbic`, `Rsquared`) and the difference between the
standard errors each function returns: [reference/regression.md](reference/regression.md).

## Models with uncertainty (`pycse.sklearn`)

All estimators follow the sklearn API (`fit`, `predict`, `score`, `get_params`) and work in a
`Pipeline`. Regressors with uncertainty support `predict(X, return_std=True)`, which returns
`(y_mean, y_std)`.

```python
from pycse.sklearn import DPOSE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import jax

model = make_pipeline(
    StandardScaler(),
    DPOSE(layers=(X.shape[1], 20, 32), activation=jax.nn.tanh),  # last entry = ensemble size
)
model.fit(X, y, dpose__maxiter=500)  # training options go to fit(), not __init__
y_mean, y_std = model.predict(X_test, return_std=True)
```

| Need | Model |
|---|---|
| Flexible UQ, calibrated per-sample std | `DPOSE` (shallow ensemble), `KfoldNN` |
| Interpretable / symbolic | `KAN`, `KANLLPR`, `SISSO`, `SISSOEnsemble` |
| Physical constraints | `JAXMonotonicRegressor`, `JAXPeriodicRegressor`, `JAXICNNRegressor` (convex) |
| Wrap a sklearn MLP | `NNBR` (Bayesian ridge head), `NNGMM` (mixture head) |
| Piecewise models | `LeafModelRegressor` |
| Linear baseline | `LinearRegressionUQ` |
| Energy landscapes, critical points, heterogeneous data | `ZENNRegressor`, `ZENNClassifier` |

Constructor arguments, examples and pitfalls for each model:
[reference/sklearn-models.md](reference/sklearn-models.md).

## Design of experiments and active learning

`LatinSquare` (three factors, equal numbers of levels, analyzed with ANOVA),
`SurfaceResponse` (Box-Behnken or central composite design plus a quadratic fit) and
`ActiveLearner` (choose the next experiments with an acquisition function):
[reference/doe.md](reference/doe.md).

## Other utilities

- `pycse.ivp(f, tspan, y0, *args)`: `scipy.integrate.solve_ivp` with sensible defaults
  (`t_eval=tspan`, `dense_output=True`, `max_step` from the `tspan` spacing). `f(t, y, *args)`.
- `pycse.feq`, `flt`, `fgt`, `fle`, `fge`: float comparisons with a tolerance.
- `pycse.hashcache.HashCache` (also `SqlCache`, `JsonCache`): decorate an expensive function
  to cache its results on disk, keyed by its bytecode and arguments.
- `pycse.supyrvisor`: `supervisor`/`manager` decorators that rerun a function after a
  `check_result` or `check_exception` function fixes its arguments.
- `pycse.pyroxy.Surrogate`: replaces an expensive function with a model; when the predicted
  error is above `tol` it calls the real function and retrains.
- `pycse.beginner`: `nsolve`, `integrate`, `first`/`last`/`rest` for teaching.

## CLI and MCP server

```bash
pycse skill install    # install this skill into ~/.claude/skills/pycse
pycse skill uninstall
pycse mcp install      # register the MCP server with Claude Desktop (needs pycse[mcp])
pycse mcp uninstall
pycse launch           # Jupyter Lab in Docker
```

The MCP server (`pycse_mcp`, stdio) provides design and analysis tools (`design_latin_square`,
`analyze_latin_square`, `design_sr`, `analyze_sr`, `sr_parity`) and lookup tools for pycse's
own documentation (`pycse_help`, `search_functions`, `get_pydoc_help`, `get_function_source`).
