"""Model-agnostic active learning for sklearn estimators with uncertainty quantification.

This module provides an active learning framework that works with any sklearn-compatible
estimator providing ``predict(X, return_std=True)``. It supports iterative experiment
selection for optimization, uncertainty reduction, and model discrimination.

Compatible models include DPOSE, KfoldNN, NNBR, LinearRegressionUQ, and any estimator
following the ``predict(X, return_std=True)`` convention.

Example
-------
>>> from pycse.sklearn import DPOSE
>>> from pycse.sklearn.active_learning import ActiveLearner, ExpectedImprovement
>>>
>>> learner = ActiveLearner(
...     model=DPOSE(),
...     bounds=[(0, 1), (0, 1)],
...     acquisition=ExpectedImprovement(minimize=True),
...     X_init=X_init,
...     y_init=y_init,  # model is fitted on these if it is not fitted yet
... )
>>> result = learner.suggest(n_points=5)
>>> # Run experiments at result.points, get y_new
>>> learner.update(result.points, y_new)  # refits on X_init + all new data

Optimization direction
----------------------
The default convention in this module is **minimization**: ``ExpectedImprovement``,
``ProbabilityOfImprovement`` and ``ThompsonSampling`` default to ``minimize=True``
and ``ActiveLearner.best_y`` is the smallest observed value. Pass ``minimize=False``
to those acquisitions (and/or to ``ActiveLearner``) to maximize. The incumbent used
by EI/PI follows each acquisition's own ``minimize`` flag.

Note that :class:`pycse.pyroxy.ActiveSurrogate` uses the opposite convention: its
``ei``/``pi`` acquisitions *maximize* (they improve on ``y_train.max()``).
"""

import abc
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass
class AcquisitionResult:
    """Result from an acquisition function evaluation.

    Attributes
    ----------
    points : np.ndarray, shape (n_points, n_features)
        Selected candidate points.
    scores : np.ndarray, shape (n_points,)
        Acquisition scores for the selected points (higher = better).
    acquisition : str
        Name of the acquisition function used.
    metadata : dict
        Additional information (batch_strategy, etc.).
    """

    points: np.ndarray
    scores: np.ndarray
    acquisition: str
    metadata: dict = field(default_factory=dict)


class AcquisitionFunction(abc.ABC):
    """Base class for acquisition functions.

    Subclasses must implement ``score(X_candidates, model)`` returning an array
    where higher values indicate more desirable points.

    Acquisition functions can be combined using arithmetic operators::

        combined = 0.7 * UCB(kappa=2) + 0.3 * PredictionVariance()
    """

    # Set by ActiveLearner before calling score()
    _y_best = None

    @abc.abstractmethod
    def score(self, X_candidates, model):
        """Score candidate points.

        Parameters
        ----------
        X_candidates : np.ndarray, shape (n_candidates, n_features)
            Candidate points to evaluate.
        model : estimator
            Fitted model with ``predict(X, return_std=True)``.

        Returns
        -------
        scores : np.ndarray, shape (n_candidates,)
            Acquisition scores. Higher is better.
        """

    @property
    def name(self):
        return self.__class__.__name__

    def __add__(self, other):
        if isinstance(other, AcquisitionFunction):
            return Composite(functions=[self, other], weights=[1.0, 1.0])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, AcquisitionFunction):
            return Composite(functions=[other, self], weights=[1.0, 1.0])
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Composite(functions=[self], weights=[float(scalar)])
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


class Composite(AcquisitionFunction):
    """Weighted combination of acquisition functions with min-max normalization.

    Each component is normalized to [0, 1] before applying weights, so that
    different acquisition functions are on comparable scales.

    Parameters
    ----------
    functions : list of AcquisitionFunction
        Component acquisition functions.
    weights : list of float
        Weights for each component.
    """

    def __init__(self, functions, weights):
        self.functions = list(functions)
        self.weights = list(weights)

    @property
    def name(self):
        parts = []
        for w, f in zip(self.weights, self.functions):
            parts.append(f"{w:.2g}*{f.name}")
        return " + ".join(parts)

    def score(self, X_candidates, model):
        total = np.zeros(len(X_candidates))
        for w, func in zip(self.weights, self.functions):
            # Components whose incumbent was set by ActiveLearner (according to
            # their own direction) keep it; otherwise inherit the composite's.
            if func._y_best is None:
                func._y_best = self._y_best
            raw = func.score(X_candidates, model)
            # Min-max normalize
            rmin, rmax = raw.min(), raw.max()
            if rmax - rmin > 0:
                normalized = (raw - rmin) / (rmax - rmin)
            else:
                normalized = np.zeros_like(raw)
            total += w * normalized
        return total

    def __add__(self, other):
        if isinstance(other, Composite):
            return Composite(
                functions=self.functions + other.functions,
                weights=self.weights + other.weights,
            )
        if isinstance(other, AcquisitionFunction):
            return Composite(
                functions=self.functions + [other],
                weights=self.weights + [1.0],
            )
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, AcquisitionFunction):
            return Composite(
                functions=[other] + self.functions,
                weights=[1.0] + self.weights,
            )
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Composite(
                functions=self.functions,
                weights=[w * scalar for w in self.weights],
            )
        return NotImplemented


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


class PredictionVariance(AcquisitionFunction):
    """Select points with highest predicted variance (pure exploration).

    score(x) = sigma(x)^2
    """

    def score(self, X_candidates, model):
        _, std = model.predict(X_candidates, return_std=True)
        return std**2


class UCB(AcquisitionFunction):
    """Upper Confidence Bound - maximize with exploration.

    score(x) = mu(x) + kappa * sigma(x)

    Parameters
    ----------
    kappa : float, default=2.0
        Exploration-exploitation trade-off. Higher = more exploration.
    """

    def __init__(self, kappa=2.0):
        self.kappa = kappa

    def score(self, X_candidates, model):
        mu, std = model.predict(X_candidates, return_std=True)
        return mu + self.kappa * std


class LCB(AcquisitionFunction):
    """Lower Confidence Bound - minimize with exploration.

    score(x) = -mu(x) + kappa * sigma(x)

    Parameters
    ----------
    kappa : float, default=2.0
        Exploration-exploitation trade-off. Higher = more exploration.
    """

    def __init__(self, kappa=2.0):
        self.kappa = kappa

    def score(self, X_candidates, model):
        mu, std = model.predict(X_candidates, return_std=True)
        return -mu + self.kappa * std


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function.

    Parameters
    ----------
    xi : float, default=0.01
        Jitter for exploration. Higher = more exploration.
    minimize : bool, default=True
        If True, seek improvements below y_best. If False, above.
    y_best : float or None, default=None
        Override the best observed value. If None, ActiveLearner sets
        it from training data: ``min(y_train)`` if ``minimize`` else
        ``max(y_train)``.
    """

    def __init__(self, xi=0.01, minimize=True, y_best=None):
        self.xi = xi
        self.minimize = minimize
        self.y_best = y_best

    def score(self, X_candidates, model):
        mu, std = model.predict(X_candidates, return_std=True)
        y_best = self.y_best if self.y_best is not None else self._y_best
        if y_best is None:
            return std  # fallback to exploration

        if self.minimize:
            imp = y_best - mu - self.xi
        else:
            imp = mu - y_best - self.xi

        # Avoid division by zero
        mask = std > 1e-10
        ei = np.zeros_like(mu)
        Z = np.zeros_like(mu)
        Z[mask] = imp[mask] / std[mask]
        ei[mask] = imp[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])
        return ei


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function.

    Parameters
    ----------
    xi : float, default=0.01
        Jitter for exploration.
    minimize : bool, default=True
        If True, seek improvements below y_best. If False, above.
    y_best : float or None, default=None
        Override the best observed value. If None, ActiveLearner sets
        it from training data: ``min(y_train)`` if ``minimize`` else
        ``max(y_train)``.
    """

    def __init__(self, xi=0.01, minimize=True, y_best=None):
        self.xi = xi
        self.minimize = minimize
        self.y_best = y_best

    def score(self, X_candidates, model):
        mu, std = model.predict(X_candidates, return_std=True)
        y_best = self.y_best if self.y_best is not None else self._y_best
        if y_best is None:
            return std

        if self.minimize:
            Z = (y_best - mu - self.xi) / np.maximum(std, 1e-10)
        else:
            Z = (mu - y_best - self.xi) / np.maximum(std, 1e-10)

        return norm.cdf(Z)


class ModelMin(AcquisitionFunction):
    """Pure exploitation: minimize predicted mean.

    score(x) = -mu(x)
    """

    def score(self, X_candidates, model):
        mu, _ = model.predict(X_candidates, return_std=True)
        return -mu


class ModelMax(AcquisitionFunction):
    """Pure exploitation: maximize predicted mean.

    score(x) = mu(x)
    """

    def score(self, X_candidates, model):
        mu, _ = model.predict(X_candidates, return_std=True)
        return mu


class EnsembleDisagreement(AcquisitionFunction):
    """Select points where ensemble members disagree most.

    Requires the model to have a ``predict_ensemble(X)`` method returning
    an array of shape ``(n_samples, n_ensemble)``.

    score(x) = std across ensemble members
    """

    def score(self, X_candidates, model):
        if not hasattr(model, "predict_ensemble"):
            raise AttributeError(
                "Model must have a predict_ensemble(X) method "
                "returning shape (n_samples, n_ensemble)."
            )
        preds = np.asarray(model.predict_ensemble(X_candidates))
        # preds shape: (n_samples, n_ensemble)
        if preds.ndim == 1:
            return np.zeros(len(X_candidates))
        return np.std(preds, axis=1)


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.

    Draws a sample from the predictive distribution at each candidate and
    scores by that sample. For batch selection, each call produces a different
    draw, giving natural diversity.

    Parameters
    ----------
    minimize : bool, default=True
        If True, lower sampled values get higher scores.
    random_state : int, np.random.Generator, np.random.RandomState or None
        Seed for reproducibility. The generator is built once, so successive
        calls to ``score`` give different draws, while two instances with the
        same integer seed give the same sequence of draws. Assigning a new
        ``random_state`` resets the generator.
    """

    def __init__(self, minimize=True, random_state=None):
        self.minimize = minimize
        self.random_state = random_state

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._random_state = value
        self._rng = _check_rng(value)

    def score(self, X_candidates, model):
        mu, std = model.predict(X_candidates, return_std=True)
        samples = self._rng.normal(mu, np.maximum(std, 1e-10))
        if self.minimize:
            return -samples
        return samples


# ---------------------------------------------------------------------------
# ActiveLearner
# ---------------------------------------------------------------------------


def _check_rng(random_state):
    """Return a ``np.random.Generator`` from a seed, Generator or RandomState."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        # Legacy RandomState: derive a Generator seed from it (reproducible)
        return np.random.default_rng(random_state.randint(0, 2**31 - 1))
    return np.random.default_rng(random_state)


def _acquisition_direction(acquisition):
    """Return the optimization direction declared by an acquisition function.

    Returns True (minimize), False (maximize) or None if the acquisition does
    not declare a direction (e.g. PredictionVariance, UCB) or its components
    disagree.
    """
    if isinstance(acquisition, Composite):
        dirs = {_acquisition_direction(f) for f in acquisition.functions} - {None}
        return dirs.pop() if len(dirs) == 1 else None
    minimize = getattr(acquisition, "minimize", None)
    return None if minimize is None else bool(minimize)


def _set_incumbents(acquisition, y, default_minimize):
    """Set ``_y_best`` on an acquisition (and its components) from data ``y``.

    Each function uses its own ``minimize`` flag when it has one, otherwise
    ``default_minimize``.
    """
    if isinstance(acquisition, Composite):
        for f in acquisition.functions:
            _set_incumbents(f, y, default_minimize)
    minimize = getattr(acquisition, "minimize", None)
    if minimize is None:
        minimize = default_minimize
    acquisition._y_best = float(np.min(y) if minimize else np.max(y))


def _generate_candidates(bounds, n_candidates, method, rng):
    """Generate candidate points within bounds.

    Parameters
    ----------
    bounds : list of (low, high) tuples
        Parameter bounds.
    n_candidates : int
        Number of candidates to generate.
    method : str
        One of "lhs", "sobol", "halton", "random".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    candidates : np.ndarray, shape (n_candidates, n_features)
    """
    from scipy.stats import qmc

    d = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    if method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=int(rng.integers(0, 2**31)))
        unit = sampler.random(n=n_candidates)
    elif method == "sobol":
        sampler = qmc.Sobol(d=d, seed=int(rng.integers(0, 2**31)))
        unit = sampler.random(n=n_candidates)
    elif method == "halton":
        sampler = qmc.Halton(d=d, seed=int(rng.integers(0, 2**31)))
        unit = sampler.random(n=n_candidates)
    elif method == "random":
        unit = rng.uniform(size=(n_candidates, d))
    else:
        raise ValueError(
            f"Unknown candidate_method {method!r}. Choose from 'lhs', 'sobol', 'halton', 'random'."
        )

    return qmc.scale(unit, lower, upper)


class ActiveLearner:
    """Model-agnostic active learner for iterative experiment selection.

    Wraps any sklearn-compatible estimator with ``predict(X, return_std=True)``
    to suggest informative new experiments via acquisition functions.

    This is NOT an sklearn estimator. It wraps one and manages the active
    learning loop (suggest → experiment → update → suggest).

    Parameters
    ----------
    model : estimator
        An sklearn-compatible model. Must support
        ``predict(X, return_std=True)`` returning ``(y_pred, y_std)``.
        Must support ``fit(X, y)`` if ``update(refit=True)`` is used or if it
        is not yet fitted when ``X_init``/``y_init`` are given.
    bounds : list of (low, high) tuples
        Parameter bounds for each feature dimension.
    acquisition : AcquisitionFunction
        Acquisition function to guide point selection.
    n_candidates : int, default=1000
        Number of candidate points to generate for each suggestion.
    candidate_method : str, default="lhs"
        Method for generating candidates: "lhs", "sobol", "halton", "random".
    random_state : int, np.random.Generator or None, default=None
        Seed for candidate generation and Thompson batches (reproducible).
    X_init, y_init : array-like or None, default=None
        Data the model is (or should be) trained on. They seed ``X_train`` /
        ``y_train`` so that ``update(refit=True)`` refits on the initial data
        *plus* all new observations. If the model is not fitted yet, it is fitted
        on them at construction.
    minimize : bool or None, default=None
        Optimization direction used for ``best_y`` / ``best_X``, for the
        incumbent of acquisitions without their own ``minimize`` flag and for
        Thompson batches of non-Thompson acquisitions. If None, it is taken from
        the acquisition's ``minimize`` attribute (``ExpectedImprovement``,
        ``ProbabilityOfImprovement``, ``ThompsonSampling``, or a ``Composite``
        whose such components agree) and otherwise defaults to True. Pass
        ``minimize=False`` explicitly when maximizing with ``UCB``/``ModelMax``.
        EI/PI always use an incumbent consistent with their own ``minimize``.

    Attributes
    ----------
    X_train : np.ndarray or None
        Accumulated training inputs (including ``X_init``).
    y_train : np.ndarray or None
        Accumulated training targets (including ``y_init``).
    iteration : int
        Number of update cycles completed.

    Examples
    --------
    >>> learner = ActiveLearner(
    ...     model=model,
    ...     bounds=[(0, 1), (0, 1)],
    ...     acquisition=ExpectedImprovement(minimize=True),
    ...     X_init=X_init,
    ...     y_init=y_init,
    ... )
    >>> result = learner.suggest(n_points=5)
    >>> y_new = run_experiment(result.points)
    >>> learner.update(result.points, y_new)
    """

    def __init__(
        self,
        model,
        bounds,
        acquisition,
        n_candidates=1000,
        candidate_method="lhs",
        random_state=None,
        X_init=None,
        y_init=None,
        minimize=None,
    ):
        self.model = model
        self.bounds = list(bounds)
        self.acquisition = acquisition
        self.n_candidates = n_candidates
        self.candidate_method = candidate_method
        self.random_state = random_state
        self.minimize = minimize

        acq_dir = _acquisition_direction(acquisition)
        if minimize is not None and acq_dir is not None and bool(minimize) != acq_dir:
            warnings.warn(
                f"ActiveLearner(minimize={minimize}) disagrees with the acquisition "
                f"({acquisition.name}, minimize={acq_dir}). The acquisition's direction "
                "is used for its incumbent; best_y/best_X use the learner's.",
                UserWarning,
                stacklevel=2,
            )

        self.X_train = None
        self.y_train = None
        self.iteration = 0
        self._rng = _check_rng(random_state)

        if (X_init is None) != (y_init is None):
            raise ValueError("X_init and y_init must be given together.")
        if X_init is not None:
            X_init = np.atleast_2d(np.asarray(X_init, dtype=float))
            y_init = np.atleast_1d(np.asarray(y_init, dtype=float)).ravel()
            if len(X_init) != len(y_init):
                raise ValueError(
                    f"X_init and y_init have inconsistent lengths: {len(X_init)} != {len(y_init)}"
                )
            self.X_train = X_init.copy()
            self.y_train = y_init.copy()
            if not _is_fitted(model):
                self.model.fit(self.X_train, self.y_train)

    @property
    def minimize_(self):
        """Resolved optimization direction (True = minimize)."""
        if self.minimize is not None:
            return bool(self.minimize)
        acq_dir = _acquisition_direction(self.acquisition)
        return True if acq_dir is None else acq_dir

    @property
    def n_observations(self):
        """Number of accumulated training observations."""
        if self.y_train is None:
            return 0
        return len(self.y_train)

    @property
    def best_y(self):
        """Best observed target value (minimum, or maximum if not ``minimize_``)."""
        if self.y_train is None:
            return None
        return float(np.min(self.y_train) if self.minimize_ else np.max(self.y_train))

    @property
    def best_X(self):
        """Input corresponding to the best observed target value."""
        if self.y_train is None:
            return None
        idx = np.argmin(self.y_train) if self.minimize_ else np.argmax(self.y_train)
        return self.X_train[idx]

    def get_params(self):
        """Get parameters of the active learner (not the wrapped model)."""
        return {
            "n_candidates": self.n_candidates,
            "candidate_method": self.candidate_method,
            "random_state": self.random_state,
            "bounds": self.bounds,
            "minimize": self.minimize,
        }

    def set_params(self, **params):
        """Set parameters of the active learner."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key!r}")
            setattr(self, key, value)
            if key == "random_state":
                self._rng = _check_rng(value)
        return self

    def suggest(self, n_points=5, batch_strategy="greedy", candidates=None, penalty_scale=0.1):
        """Suggest the next points to evaluate.

        Parameters
        ----------
        n_points : int, default=5
            Number of points to suggest.
        batch_strategy : str, default="greedy"
            Strategy for batch selection:
            - "greedy": Top-k by acquisition score.
            - "penalized": Sequential greedy with distance penalty for diversity.
            - "thompson": Re-sample acquisition for each batch point.
        candidates : np.ndarray or None, default=None
            Custom candidate points. If None, generates candidates using
            ``candidate_method``.
        penalty_scale : float, default=0.1
            Width of the Gaussian penalty used by the "penalized" strategy, as a
            fraction of each dimension's range. Larger values spread a batch
            out more.

        Returns
        -------
        result : AcquisitionResult
            Selected points, scores, and metadata.
        """
        if candidates is None:
            candidates = _generate_candidates(
                self.bounds, self.n_candidates, self.candidate_method, self._rng
            )

        candidates = np.asarray(candidates)

        # Set the incumbent (y_best) on the acquisition from training data,
        # following each acquisition's own direction (min or max).
        if self.y_train is not None and len(self.y_train) > 0:
            _set_incumbents(self.acquisition, self.y_train, self.minimize_)

        if batch_strategy == "greedy":
            return self._suggest_greedy(candidates, n_points)
        elif batch_strategy == "penalized":
            return self._suggest_penalized(candidates, n_points, penalty_scale)
        elif batch_strategy == "thompson":
            return self._suggest_thompson(candidates, n_points)
        else:
            raise ValueError(
                f"Unknown batch_strategy {batch_strategy!r}. "
                "Choose from 'greedy', 'penalized', 'thompson'."
            )

    def _suggest_greedy(self, candidates, n_points):
        """Top-k by acquisition score."""
        scores = self.acquisition.score(candidates, self.model)
        n_points = min(n_points, len(candidates))
        top_idx = np.argsort(scores)[-n_points:][::-1]
        return AcquisitionResult(
            points=candidates[top_idx],
            scores=scores[top_idx],
            acquisition=self.acquisition.name,
            metadata={"batch_strategy": "greedy"},
        )

    def _suggest_penalized(self, candidates, n_points, penalty_scale=0.1):
        """Sequential greedy with a multiplicative Gaussian distance penalty.

        Scores are shifted to be non-negative, and each selected point
        multiplies the scores of its neighbors by ``1 - exp(-d**2 / 2 s**2)``
        (``d`` is the distance in range-normalized units and ``s`` is
        ``penalty_scale``). Selected points are masked so they are never chosen
        twice.
        """
        scores = np.asarray(self.acquisition.score(candidates, self.model), dtype=float)
        if penalty_scale <= 0:
            raise ValueError("penalty_scale must be positive.")
        selected_idx = []

        ranges = np.array([b[1] - b[0] for b in self.bounds], dtype=float)
        ranges[ranges == 0] = 1.0

        finite = np.isfinite(scores)
        base = scores - (scores[finite].min() if finite.any() else 0.0)
        # A small offset keeps ties broken by the score when all shifted
        # scores near a selection are zero.
        base = np.where(finite, base, 0.0) + 1e-12
        weight = np.ones(len(candidates))
        available = np.ones(len(candidates), dtype=bool)

        for _ in range(min(n_points, len(candidates))):
            penalized_scores = np.where(available, base * weight, -np.inf)
            idx = int(np.argmax(penalized_scores))
            selected_idx.append(idx)
            available[idx] = False

            # Down-weight the neighbors of the selected point
            dists = np.linalg.norm((candidates - candidates[idx]) / ranges, axis=1)
            weight *= 1.0 - np.exp(-0.5 * (dists / penalty_scale) ** 2)

        selected_idx = np.array(selected_idx)
        return AcquisitionResult(
            points=candidates[selected_idx],
            scores=scores[selected_idx],
            acquisition=self.acquisition.name,
            metadata={"batch_strategy": "penalized"},
        )

    def _suggest_thompson(self, candidates, n_points):
        """Thompson sampling: re-draw for each batch point."""
        mu, std = self.model.predict(candidates, return_std=True)
        selected_idx = []
        selected_scores = []

        for _ in range(min(n_points, len(candidates))):
            samples = self._rng.normal(mu, np.maximum(std, 1e-10))
            # Use the acquisition's minimize preference if it's ThompsonSampling,
            # otherwise the learner's direction (default: minimize)
            if isinstance(self.acquisition, ThompsonSampling):
                minimize = self.acquisition.minimize
            else:
                minimize = self.minimize_
            scores = -samples if minimize else samples

            idx = np.argmax(scores)
            selected_idx.append(idx)
            selected_scores.append(scores[idx])

        selected_idx = np.array(selected_idx)
        selected_scores = np.array(selected_scores)
        return AcquisitionResult(
            points=candidates[selected_idx],
            scores=selected_scores,
            acquisition=self.acquisition.name,
            metadata={"batch_strategy": "thompson"},
        )

    def update(self, X_new, y_new, refit=True):
        """Add new observations and optionally refit the model.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, n_features) or (n_features,)
            New input points.
        y_new : np.ndarray, shape (n_new,) or scalar
            New target values.
        refit : bool, default=True
            If True, refit the model on all accumulated data (``X_init``/``y_init``
            plus every point passed to ``update``).
        """
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new).ravel()

        if self.X_train is None and refit and _is_fitted(self.model):
            warnings.warn(
                "ActiveLearner does not know the data the model was fitted on, so "
                "update(refit=True) refits on the new points only. Pass X_init/y_init "
                "to ActiveLearner to keep the initial data.",
                UserWarning,
                stacklevel=2,
            )

        if self.X_train is None:
            self.X_train = X_new.copy()
            self.y_train = y_new.copy()
        else:
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.concatenate([self.y_train, y_new])

        self.iteration += 1

        if refit:
            self.model.fit(self.X_train, self.y_train)


def _is_fitted(model):
    """Best-effort check whether ``model`` has been fitted.

    Mirrors sklearn's ``check_is_fitted`` convention (``__sklearn_is_fitted__``
    or public attributes ending in ``_``) without requiring a BaseEstimator.
    """
    if hasattr(model, "__sklearn_is_fitted__"):
        try:
            return bool(model.__sklearn_is_fitted__())
        except Exception:
            return False
    try:
        attrs = vars(model)
    except TypeError:
        return False
    return any(k.endswith("_") and not k.startswith("__") for k in attrs)
