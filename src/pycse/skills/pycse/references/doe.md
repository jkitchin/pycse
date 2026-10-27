# Design of experiments and active learning

## Contents
- Latin square: `LatinSquare`
- Surface response: `SurfaceResponse`
- Active learning: `ActiveLearner` and acquisition functions

## Latin square (`pycse.sklearn.lhc.LatinSquare`)

Exactly three factors with the same number of levels `n` (numeric or categorical), studied in
`n*n` runs instead of a full `n**3` factorial. Each level of each factor appears once with each
level of the others, so main effects can be separated with ANOVA. It does not estimate
interactions.

```python
import pandas as pd
from pycse.sklearn.lhc import LatinSquare

ls = LatinSquare(
    {"Temperature": [100, 150, 200], "Pressure": [1, 2, 3], "Catalyst": ["A", "B", "C"]},
    random_state=0,
)
design = ls.design(shuffle=True)  # DataFrame with 9 runs
# ... run the experiments in this order ...
y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Yield")
ls.fit(design, y)
print(ls.anova())  # which factors matter
ls.predict([150, 2, "B"])
```

## Surface response (`pycse.sklearn.surface_response.SurfaceResponse`)

Builds a response-surface design and fits a quadratic model (a sklearn pipeline), for
optimizing a few continuous factors when each experiment is expensive.

```python
from pycse.sklearn.surface_response import SurfaceResponse

sr = SurfaceResponse(
    inputs=["T", "P"], outputs=["yield"], bounds=[[100, 200], [1, 5]], design="bbdesign"
)  # or "ccdesign"
design = sr.generate_design()  # DataFrame in real units
results = run_experiments(design)  # shape (n_runs, n_outputs), same row order
sr.set_results(results)
sr.fit()
print(sr.summary())  # coefficients with significance
sr.parity()  # parity plot
```

`design` is `"bbdesign"` (Box-Behnken, the default) or `"ccdesign"` (central composite).
Coefficients in `summary()` are in coded units (bounds mapped to [-1, 1]).
`set_results` must follow `generate_design`; rows match the design rows.

## Active learning (`pycse.sklearn.active_learning`)

`ActiveLearner` wraps any model with `predict(X, return_std=True)` and chooses where to run
the next experiments. It is a loop helper, not an estimator.

```python
from pycse.sklearn.active_learning import ActiveLearner, ExpectedImprovement

learner = ActiveLearner(
    model=model,
    bounds=[(0, 1), (0, 1)],
    acquisition=ExpectedImprovement(minimize=True),
    X_init=X0,
    y_init=y0,
    random_state=0,
)
for _ in range(5):
    result = learner.suggest(n_points=3, batch_strategy="penalized")
    y_new = run_experiments(result.points)
    learner.update(result.points, y_new)  # refits on all data so far
learner.best_X, learner.best_y
```

Acquisition functions: `ExpectedImprovement`, `ProbabilityOfImprovement`, `UCB`, `LCB`,
`PredictionVariance` (pure exploration), `ModelMin`, `ModelMax`, `EnsembleDisagreement`,
`ThompsonSampling`, and `Composite(functions, weights)`.
Batch strategies: `"greedy"` (top scores, may cluster), `"penalized"` (spreads the batch out),
`"thompson"`. When maximizing with `UCB` or `ModelMax`, pass `minimize=False` to `ActiveLearner`.
