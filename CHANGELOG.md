# Changelog

Notable changes to pycse. Earlier releases are described on the
[GitHub releases page](https://github.com/jkitchin/pycse/releases).

## 2.12.0 (unreleased)

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
