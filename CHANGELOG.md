# Changelog

Notable changes to pycse. Earlier releases are described on the
[GitHub releases page](https://github.com/jkitchin/pycse/releases).

## 2.13.0 (2026-10-24)

### New

- `pycse.sklearn.mip`: export trained piecewise-linear models as mixed-integer linear
  constraints in Pyomo, to use them as surrogates inside an optimization problem.
  `to_pyomo(input_bounds=..., block=..., inputs=...)` is available on `DPOSE`,
  `LLPRRegressor`, `NNBR`, `JAXICNNRegressor` and `JAXMonotonicRegressor` (with the ReLU
  activation), `KAN` and `KANLLPR` (with `spline_order=1, base_activation="linear"`),
  `LeafModelRegressor` (with a linear leaf model) and `LinearRegressionUQ`. Models with
  uncertainty also export the predictive variance as a second-order cone, so `mean + k * std`
  can be optimized with a MISOCP solver. `to_pyomo(formulation="convex")` exports an
  input-convex network (ICNN, or `JAXMonotonicRegressor(convexity="convex")`) as an LP
  without binaries, for minimization. Install with `pip install "pycse[mip]"`.
  - `include_std=True` adds `block.std` as a second-order cone (exact when the objective
    penalizes the std, e.g. robust design); `include_std="exact"` makes it an equality for
    objectives that reward it (e.g. exploration).
  - `JAXPeriodicRegressor(activation="relu")` exports approximately: its sin/cos features
    are interpolated (`to_pyomo(points_per_period=...)`), with a bound on the error
    (`mip_network().max_feature_error(bounds)`).
  - `SISSO` and `SISSOEnsemble` export their equations as exact nonlinear Pyomo
    expressions, for a global MINLP solver.
  - `LeafModelRegressor.to_pyomo(split_gap=...)` keeps points on a split in the left leaf.
  - A new page in the documentation, "Optimizing over machine-learned models with MIP",
    walks through all of these.
- `KfoldNN(activation=...)` chooses the hidden activation (`"swish"`, the default,
  `"relu"` or `"tanh"`).
- `LLPRRegressor` has new options: `calibration` (`"auto"`, `"holdout"` or `"cv"`) with `cv`
  folds, `standardize`, `warm_start` for refitting in an active learning loop, and `verbose`.
  ReLU and tanh networks export the mean prediction with `get_dense_layers()`, `to_onnx()`
  and `to_omlt()` (these need `onnx` and `omlt`, which are not installed by any extra).

### Changed

- `LLPRRegressor` gives different results with the same code:
  - It standardizes X and y internally (`standardize=True`); predictions, uncertainties and
    `alpha_squared_` are in the original units of y.
  - The last-layer features include a constant, so the output bias contributes to the
    uncertainty and the std is never zero (before, a point where every last-layer ReLU was
    off got σ = 0).
  - With fewer than 100 samples, `calibration="auto"` calibrates on K-fold out-of-fold
    residuals and trains the final model on all the data, so a fit trains `cv + 1` networks.
    α² is the NLL-optimal value in closed form, and ζ² is searched relative to
    trace(FᵀF) / n_features.
  - The last partial batch is used, so datasets smaller than `batch_size` train.
  - It no longer prints during `fit()` unless `verbose=True`.
  - The predictive variance is no longer recompiled on every call, so `predict()` is faster.
- Dependency updates in `uv.lock`: certifi, filelock, mcp, nbconvert, pillow 13 and
  setuptools.

- `KAN.to_pyomo()` uses the shared MIP module. The exported model is exact outside the spline
  grid (the grid clipping was not represented before), needs one set of segment binaries per
  input instead of per edge, and supports `n_ensemble > 1` (the ensemble mean, with its std).
  Without `input_bounds`, the inputs are bounded by the range the spline grid covers in data
  units instead of by `grid_range` itself.

## 2.12.0 (2026-10-15)

### Breaking changes

- **Python 3.12 or newer is required.** The minimum versions of the JAX stack are now
  `jax>=0.10`, `flax>=0.12.7` and `optax>=0.2.5`.
- **Prediction intervals from `predict()` and `polyval()` have shape `(n, 2)`** instead of
  `(2, n)`, the same as `nlpredict()`. `yint[:, 0]` is the lower bound and `yint[:, 1]` the
  upper bound. Code that unpacks `lower, upper = yint` must change to `yint.T`.
- **`optimizer="bfgs"` and `"lbfgs"` run real L-BFGS** (`optax.lbfgs` with a line search).
  Before, they ran Adam with a cosine schedule, so fits are different (and converged). Use
  `optimizer="adam_cosine"` to get the old behavior. This affects the JAX models that take
  `optimizer=`: `DPOSE`, `KAN`, `KANLLPR`, `KfoldNN` and `SISSOEnsemble`.
- **`JAXMonotonicRegressor` is monotone only by default.** It used to be convex as well,
  so concave or S-shaped monotone data could not be fitted. Pass `convexity="convex"` for the
  old behavior; `"concave"` is also available. (#98)

### New

- `pycse.sklearn.zenn`: `ZENNRegressor`, `ZENNRegressorNLL` and `ZENNClassifier`, an
  independent JAX/Flax implementation of Zentropy-Enhanced Neural Networks (Wang et al.,
  PNAS 2026).
- `LeafModelRegressor(refine_splits=True)` chooses each split threshold for the leaf model
  instead of for constant leaves, which places breakpoints correctly for piecewise-linear
  data. `get_params()` now includes the decision-tree parameters, so `clone()` keeps them.
  (#99)
- `ActiveLearner.suggest(penalty_scale=...)` sets the width of the penalty for
  `batch_strategy="penalized"`.
- `ActiveSurrogate.build(random_state=...)` makes surrogate builds reproducible.
- `LatinSquare(random_state=...)` seeds the shuffling without touching NumPy's global state.
- The MCP server works with `mcp` 2.x (`MCPServer`) as well as 1.x (`FastMCP`).
- The Claude Code skill (`pycse skill install`) has been rewritten. It now covers the
  regression functions, all the sklearn models, DOE and active learning, using reference
  files that are installed with it.

### Fixed

- `pycse.__version__` reports the installed version. It had been hardcoded as `2.3.5`
  since before 2.10.0.
- `nlpredict()` works for a single prediction point.
- `NNGMM` computes the conditional standard deviation itself, so it works on Python 3.12+.
- `NNBR` reports the correct standard deviation of the intercept.
- `ActiveLearner`: penalized batches no longer contain duplicate points, and
  `ThompsonSampling` gives a new draw on each call instead of reusing the same one. (#95)
- `pyroxy`: a `Surrogate` whose model has not been fitted calls the real function instead of
  answering from the model's prior (e.g. a `GaussianProcessRegressor`). (#96)
- `hashcache`: `SqlCache` and `JsonCache` return NumPy arrays, NumPy scalars and tuples with
  their original types. `HashCache.dump()`/`load()` use `HashCache.cache` instead of
  `./cache`. (#94)
- `supyrvisor`: functions decorated with `check_result`/`check_exception` keep their names in
  logs, tracebacks and `help()`. (#97)
- `pycse.sklearn` imports models lazily, so importing it no longer loads every backend.
- Fixes in `beginner`, `utils` and the CLI.

### Deprecated

- CLI: `pycse mcp install-mcp`, `pycse skill install-skill` (and the matching `uninstall-*`
  commands) are hidden aliases for `pycse mcp install`, `pycse skill install`, and so on.
- `SurfaceResponse.design()` is deprecated in favor of `generate_design()`, and
  `SurfaceResponse.set_output(data)` in favor of `set_results(data)`.

### Development

- Slow tests run on release tags and manual dispatch instead of every push to master.
- The tests in `src/pycse/sklearn/tests` (active learning and ZENN) are now collected;
  18 stale scripts there that could not run were removed.
- `ruff` is pinned to 0.16.7 in CI and pre-commit.
- The publish workflow builds with Python 3.12.
- A JOSS paper draft is in `joss/`.
