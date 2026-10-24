# sklearn-compatible models (`pycse.sklearn`)

Import from `pycse.sklearn` (imports are lazy, so JAX loads only when a JAX model is used):
`from pycse.sklearn import DPOSE, KAN, LLPRRegressor, ...`.

## Contents
- Common conventions
- Neural-network ensembles: `DPOSE`, `KfoldNN`
- Last-layer uncertainty: `LLPRRegressor`, `KANLLPR`, constrained networks
- Interpretable models: `KAN`, `SISSO`, `SISSOEnsemble`
- Wrappers around sklearn models: `NNBR`, `NNGMM`, `LeafModelRegressor`, `LinearRegressionUQ`
- `ZENNRegressor`, `ZENNClassifier`
- MIP export (`to_pyomo`)
- Optimizers

## Common conventions

- `predict(X, return_std=True)` returns `(y_mean, y_std)` on every regressor below.
- Scale inputs. Use a `Pipeline` with `StandardScaler`; the constrained JAX networks scale
  internally (`standardize_X=True`, `standardize_y=True`).
- In a pipeline, `fit` options go through the step name: `make_pipeline(StandardScaler(), DPOSE(...))`
  names the step `dpose`, so write `pipe.fit(X, y, dpose__maxiter=500)`.
  `pipe.predict(X, return_std=True)` is passed through to the last step.
- Most models have `report()`, `plot(X, y)` and `uncertainty_metrics(X, y)` (RMSE, NLL,
  miscalibration, z-score statistics) after fitting.
- Mark tests that train these models as slow; use 30–50 samples and few iterations.

## Neural-network ensembles

**`DPOSE(layers, activation=nn.relu, seed=19, loss_type="crps", min_sigma=1e-3, optimizer="bfgs")`**
Shallow ensemble: the last entry of `layers` is the ensemble size (16–64). `activation` is a
callable such as `jax.nn.tanh`, not a string.
`fit(X, y, val_X=None, val_y=None, pretrain_with_mse=None, maxiter=..., tol=...)`.
Prefer `loss_type="crps"`; `"nll"` can inflate the uncertainty without pre-training.
`predict_ensemble(X)` returns all members, shape `(n_samples, n_ensemble)`; apply a nonlinear
function to the members and then take the mean and std to propagate uncertainty.

```python
import jax
from pycse.sklearn import DPOSE

m = DPOSE(layers=(X.shape[1], 20, 32), activation=jax.nn.tanh)
m.fit(X_scaled, y, maxiter=500)
y_mean, y_std = m.predict(X_scaled, return_std=True)
members = m.predict_ensemble(X_scaled)
```

**`KfoldNN(layers, xtrain=0.1, seed=19)`**: each of the last-layer outputs is trained on a
fraction `xtrain` of the data. A smaller `xtrain` gives wider uncertainty.

## Last-layer uncertainty (LLPR)

These models train one network, then estimate uncertainty from the last-layer features.
`alpha_squared` and `zeta_squared` are calibrated on a held-out `val_size` fraction when `"auto"`.

- **`LLPRRegressor(hidden_dims=(64, 64), activation="silu", n_epochs=400, ...)`** (alias `LLPR`).
  `calibration="auto"` uses K-fold (`cv=5`) calibration below 100 samples and a holdout otherwise;
  X and y are standardized internally (`standardize=True`). ReLU/tanh networks export the mean
  model with `to_onnx()` / `to_omlt()`.
- **`KANLLPR(layers, grid_size=5, spline_order=3, ...)`**: KAN with LLPR uncertainty.
- **`JAXMonotonicRegressor(hidden_dims=(32, 32), monotonicity=1, convexity=None, epochs=50, ...)`**:
  `monotonicity` is one value per feature: `1` increasing, `-1` decreasing, `0` unconstrained.
  `convexity` can be `"convex"` or `"concave"`; the default is monotone only.
- **`JAXPeriodicRegressor(hidden_dims=(32, 32), periodicity={0: 2*np.pi}, n_harmonics=5, learn_period=False, ...)`**:
  `periodicity` maps a feature index to its period.
- **`JAXICNNRegressor(hidden_dims=(32, 32), strong_convexity_mu=0.0, ...)`**: output is convex in
  all inputs, which suits surrogate models inside an optimizer. It has no `return_std`.

## Interpretable models

- **`KAN(layers, grid_size=5, spline_order=3, n_ensemble=1, loss_type="mse", ...)`**:
  Kolmogorov-Arnold network. Set `n_ensemble > 1` with `loss_type="crps"` for uncertainty.
  `plot_activations()` shows the learned functions. With `spline_order=1` and
  `base_activation="linear"`, `to_pyomo(input_bounds=...)` exports a MILP model (see MIP export).
- **`SISSO(operators=None, n_expansion=2, n_term=2, k=20)`**: symbolic regression; the formula is
  in `model.equation_`. Needs `pip install "pycse[sisso]"`.
- **`SISSOEnsemble(n_models=5, n_terms_range=(1, 3), loss_type="crps", ...)`**: an ensemble of
  SISSO equations with calibrated uncertainty.

## Wrappers around sklearn models

- **`NNBR(nn, br)`** (alias of `NeuralNetworkBLR`): `nn` is an `MLPRegressor`, `br` a
  `BayesianRidge`; the Bayesian ridge replaces the last layer.
- **`NNGMM(nn, n_components=1, n_samples=500)`** (alias of `NeuralNetworkGMM`): Gaussian mixture
  on the network features, for multimodal noise.
- **`LeafModelRegressor(leaf_model, refine_splits=False, **tree_kwargs)`**: a decision tree with
  a separate `leaf_model` (e.g. `LinearRegressionUQ()`) in each leaf. Set `refine_splits=True` for
  piecewise-linear data so that the breakpoints are chosen for the leaf model, not for constant
  leaves. Warns when predicting outside the training range.
- **`LinearRegressionUQ()`**: ordinary least squares with prediction standard errors.

```python
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from pycse.sklearn import NNBR

m = NNBR(MLPRegressor(hidden_layer_sizes=(20,), max_iter=2000), BayesianRidge())
m.fit(X, y)
y_mean, y_std = m.predict(X, return_std=True)
```

## ZENN

`ZENNRegressor(n_configs=6, hidden_dims=(16, 16), max_epochs=5000, loss_type="nll", ...)` and
`ZENNClassifier(n_temperatures=4, ...)` implement Zentropy-Enhanced Neural Networks
(Wang et al., PNAS 2026): several "configuration" networks are combined with a learned
temperature, which suits energy landscapes, critical points and data from several sources.
`ZENNRegressor.predict(X, T=None)` takes an optional temperature;
`predict_with_uncertainty(X)` returns the mean and std. `ZENNRegressorNLL` uses a
negative-log-likelihood head. This is an independent JAX implementation, not the authors' code.

## MIP export (`to_pyomo`)

A piecewise-linear model can be embedded in an optimization problem as mixed-integer linear
constraints with `to_pyomo()`. Needs `pip install "pycse[mip]"`. Supported configurations
(others raise `ValueError`):

| Model | Requirement | Exports std |
|---|---|---|
| `DPOSE` | `activation=nn.relu` (default) | yes |
| `LLPRRegressor` | `activation="relu"` | yes |
| `NNBR` | `MLPRegressor(activation="relu")` | yes |
| `JAXICNNRegressor` | `activation="relu"`, `strong_convexity_mu=0` | no |
| `JAXMonotonicRegressor` | `activation="relu"` | yes |
| `KAN` | `spline_order=1, base_activation="linear"` | with `n_ensemble > 1` |
| `KANLLPR` | `spline_order=1, base_activation="linear"` | yes |
| `KfoldNN` | `activation="relu"` | yes |
| `LeafModelRegressor` | linear `leaf_model` (optionally after a scaler) | no |
| `LinearRegressionUQ` | none (no binaries) | yes |
| `JAXPeriodicRegressor` | `activation="relu"`; approximate (sin/cos interpolated) | yes |
| `SISSO`, `SISSOEnsemble` | none; exact nonlinear expressions (needs a MINLP solver) | yes |

Every input needs finite bounds; tighter bounds solve faster.

```python
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.T = pyo.Var(bounds=(300, 500))
m.P = pyo.Var(bounds=(1, 10))
m.nn = pyo.Block()
model.to_pyomo(block=m.nn, inputs=[m.T, m.P])  # bounds come from the vars
m.obj = pyo.Objective(expr=m.nn.y[0], sense=pyo.maximize)
pyo.SolverFactory("highs").solve(m)
```

`m.nn.y[j]` is the prediction and `m.nn.variance[j]` the predictive variance.
`to_pyomo(include_std=True)` adds `m.nn.std[j]` as a second-order cone, exact when the
objective penalizes the std (minimize `y + 2 * std`); `include_std="exact"` is needed when
it rewards the std (minimize `y - 2 * std`). Both need a solver for quadratic constraints
such as Gurobi (`pyo.SolverFactory("gurobi_direct")`).

Model-specific options go through `to_pyomo(**options)`:
`JAXPeriodicRegressor.to_pyomo(points_per_period=32)` sets the interpolation grid of the
sin/cos features (feature error at most `(2 pi / points_per_period)**2 / 8`), and
`LeafModelRegressor.to_pyomo(split_gap=1e-4)` keeps points on a split in the left leaf.
SISSO models need a nonlinear solver, e.g. `pyo.SolverFactory("gurobi_direct_minlp")`.
`model.mip_network()` returns the layers as numpy arrays (`pycse.sklearn.mip.MIPNetwork`, or
`TreeModel` for `LeafModelRegressor`).

For an input-convex network (`JAXICNNRegressor`, or `JAXMonotonicRegressor(convexity="convex")`)
whose output is minimized or bounded above, `to_pyomo(formulation="convex")` is an LP with no
binaries.

## Optimizers

JAX models take `optimizer=`: `"bfgs"`/`"lbfgs"` (L-BFGS with a line search, the default),
`"adam"`, `"adamw"`, `"sgd"`, `"muon"`, `"adam_cosine"`, `"gradient_descent"`.
Before 2.12, `"bfgs"` actually ran Adam with a cosine schedule; use `"adam_cosine"` to
reproduce old results.
