"""Regression tests for ActiveLearner: incumbent direction, initial data and RNG.

(The broader unit tests live in src/pycse/sklearn/tests/test_active_learning.py;
this file uses a different name to avoid a module-basename clash.)
"""

import warnings

import numpy as np
import pytest

from pycse.sklearn.active_learning import (
    ActiveLearner,
    ExpectedImprovement,
    PredictionVariance,
    ProbabilityOfImprovement,
    ThompsonSampling,
    UCB,
)


class LinearMock:
    """mu = x0, constant std; records the data it is fitted on."""

    def __init__(self):
        self.fit_sizes = []

    def fit(self, X, y):
        self.fit_sizes.append(len(y))
        self.coef_ = 1.0
        return self

    def predict(self, X, return_std=False):
        X = np.atleast_2d(X)
        mu = X[:, 0]
        if return_std:
            return mu, 0.1 * np.ones(len(X))
        return mu


X0 = np.array([[0.2], [0.8], [0.5]])
Y0 = np.array([0.2, 0.8, 0.5])


# ---------------------------------------------------------------------------
# Bug 3: incumbent must follow the optimization direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("acq_cls", [ExpectedImprovement, ProbabilityOfImprovement])
def test_incumbent_is_max_when_maximizing(acq_cls):
    acq = acq_cls(minimize=False)
    learner = ActiveLearner(LinearMock(), [(0, 1)], acq, X_init=X0, y_init=Y0, random_state=0)
    learner.suggest(n_points=2)
    assert acq._y_best == 0.8
    assert learner.minimize_ is False
    assert learner.best_y == 0.8
    np.testing.assert_array_equal(learner.best_X, [0.8])


@pytest.mark.parametrize("acq_cls", [ExpectedImprovement, ProbabilityOfImprovement])
def test_incumbent_is_min_when_minimizing(acq_cls):
    acq = acq_cls(minimize=True)
    learner = ActiveLearner(LinearMock(), [(0, 1)], acq, X_init=X0, y_init=Y0, random_state=0)
    learner.suggest(n_points=2)
    assert acq._y_best == 0.2
    assert learner.best_y == 0.2
    np.testing.assert_array_equal(learner.best_X, [0.2])


def test_maximizing_ei_suggests_high_points():
    """With mu = x, maximizing EI should suggest points near x = 1."""
    learner = ActiveLearner(
        LinearMock(),
        [(0, 1)],
        ExpectedImprovement(minimize=False),
        X_init=X0,
        y_init=Y0,
        random_state=0,
    )
    result = learner.suggest(n_points=3)
    assert np.all(result.points > 0.8)


def test_composite_components_use_own_direction():
    ei = ExpectedImprovement(minimize=False)
    acq = 0.5 * ei + 0.5 * PredictionVariance()
    learner = ActiveLearner(LinearMock(), [(0, 1)], acq, X_init=X0, y_init=Y0, random_state=0)
    assert learner.minimize_ is False
    learner.suggest(n_points=2)
    assert ei._y_best == 0.8


def test_default_direction_is_minimize():
    """Acquisitions without a direction keep the historical min convention."""
    learner = ActiveLearner(LinearMock(), [(0, 1)], UCB(), X_init=X0, y_init=Y0)
    assert learner.minimize_ is True
    assert learner.best_y == 0.2


def test_explicit_learner_minimize():
    learner = ActiveLearner(LinearMock(), [(0, 1)], UCB(), X_init=X0, y_init=Y0, minimize=False)
    assert learner.best_y == 0.8
    assert learner.get_params()["minimize"] is False


def test_conflicting_direction_warns():
    with pytest.warns(UserWarning, match="disagrees"):
        ActiveLearner(LinearMock(), [(0, 1)], ExpectedImprovement(minimize=False), minimize=True)


def test_thompson_batch_follows_learner_direction():
    learner = ActiveLearner(
        LinearMock(), [(0, 1)], UCB(), X_init=X0, y_init=Y0, minimize=False, random_state=0
    )
    result = learner.suggest(n_points=3, batch_strategy="thompson")
    assert np.all(result.points > 0.7)


# ---------------------------------------------------------------------------
# Bug 4: initial-data API and data accumulation
# ---------------------------------------------------------------------------


def test_init_data_fits_unfitted_model():
    model = LinearMock()
    learner = ActiveLearner(model, [(0, 1)], PredictionVariance(), X_init=X0, y_init=Y0)
    assert model.fit_sizes == [3]
    assert learner.n_observations == 3


def test_prefitted_model_not_refit_at_init():
    model = LinearMock().fit(X0, Y0)
    ActiveLearner(model, [(0, 1)], PredictionVariance(), X_init=X0, y_init=Y0)
    assert model.fit_sizes == [3]


def test_update_accumulates_initial_data():
    model = LinearMock().fit(X0, Y0)
    learner = ActiveLearner(model, [(0, 1)], PredictionVariance(), X_init=X0, y_init=Y0)
    learner.update([[0.1]], [0.1])
    learner.update([[0.9], [0.95]], [0.9, 0.95])
    assert model.fit_sizes == [3, 4, 6]
    assert learner.n_observations == 6
    assert learner.iteration == 2


def test_update_prefitted_without_init_warns():
    model = LinearMock().fit(X0, Y0)
    learner = ActiveLearner(model, [(0, 1)], PredictionVariance())
    with pytest.warns(UserWarning, match="X_init"):
        learner.update([[0.1]], [0.1])


def test_update_unfitted_without_init_no_warning():
    learner = ActiveLearner(LinearMock(), [(0, 1)], PredictionVariance())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        learner.update([[0.1]], [0.1])


def test_init_data_validation():
    with pytest.raises(ValueError, match="together"):
        ActiveLearner(LinearMock(), [(0, 1)], PredictionVariance(), X_init=X0)
    with pytest.raises(ValueError, match="inconsistent"):
        ActiveLearner(LinearMock(), [(0, 1)], PredictionVariance(), X_init=X0, y_init=Y0[:2])


# ---------------------------------------------------------------------------
# RNG: default_rng with reproducible random_state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["lhs", "sobol", "halton", "random"])
def test_random_state_reproducible(method):
    def run():
        learner = ActiveLearner(
            LinearMock(),
            [(0, 1), (0, 2)],
            PredictionVariance(),
            n_candidates=64,
            candidate_method=method,
            random_state=123,
        )
        return learner.suggest(n_points=3, batch_strategy="thompson").points

    np.testing.assert_array_equal(run(), run())


def test_uses_generator():
    learner = ActiveLearner(LinearMock(), [(0, 1)], PredictionVariance(), random_state=1)
    assert isinstance(learner._rng, np.random.Generator)


def test_set_params_random_state_resets_rng():
    learner = ActiveLearner(LinearMock(), [(0, 1)], PredictionVariance(), random_state=1)
    p1 = learner.suggest(n_points=2).points
    learner.set_params(random_state=1)
    p2 = learner.suggest(n_points=2).points
    np.testing.assert_array_equal(p1, p2)


def test_thompson_sampling_random_state_types():
    X = np.linspace(0, 1, 10)[:, None]
    for rs in [0, np.random.default_rng(0), np.random.RandomState(0), None]:
        s = ThompsonSampling(random_state=rs).score(X, LinearMock())
        assert s.shape == (10,)
    s1 = ThompsonSampling(random_state=5).score(X, LinearMock())
    s2 = ThompsonSampling(random_state=5).score(X, LinearMock())
    np.testing.assert_array_equal(s1, s2)
