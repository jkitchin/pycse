"""Tests for the active learning module.

All tests use mock models and are fast (no ML training).
"""

import numpy as np
import pytest

from pycse.sklearn.active_learning import (
    AcquisitionResult,
    ActiveLearner,
    Composite,
    EnsembleDisagreement,
    ExpectedImprovement,
    LCB,
    ModelMax,
    ModelMin,
    PredictionVariance,
    ProbabilityOfImprovement,
    ThompsonSampling,
    UCB,
)


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------


class MockUQModel:
    """Simple mock model with predict(X, return_std=True)."""

    def __init__(self, mu_func=None, std_func=None):
        self._mu_func = mu_func or (lambda X: np.sum(X, axis=1))
        self._std_func = std_func or (lambda X: np.ones(len(X)) * 0.5)
        self._fit_count = 0

    def fit(self, X, y):
        self._fit_count += 1
        return self

    def predict(self, X, return_std=False):
        X = np.atleast_2d(X)
        mu = self._mu_func(X)
        if return_std:
            return mu, self._std_func(X)
        return mu


class MockEnsembleModel(MockUQModel):
    """Mock model that also supports predict_ensemble."""

    def __init__(self, n_ensemble=5, **kwargs):
        super().__init__(**kwargs)
        self.n_ensemble = n_ensemble

    def predict_ensemble(self, X):
        X = np.atleast_2d(X)
        mu = self._mu_func(X)
        rng = np.random.RandomState(42)
        # shape: (n_samples, n_ensemble)
        return mu[:, None] + rng.normal(0, 0.5, size=(len(X), self.n_ensemble))


# ---------------------------------------------------------------------------
# Test AcquisitionResult
# ---------------------------------------------------------------------------


class TestAcquisitionResult:
    def test_creation(self):
        r = AcquisitionResult(
            points=np.array([[1, 2]]),
            scores=np.array([0.5]),
            acquisition="UCB",
        )
        assert r.acquisition == "UCB"
        assert r.metadata == {}

    def test_metadata(self):
        r = AcquisitionResult(
            points=np.zeros((3, 2)),
            scores=np.zeros(3),
            acquisition="EI",
            metadata={"batch_strategy": "greedy"},
        )
        assert r.metadata["batch_strategy"] == "greedy"


# ---------------------------------------------------------------------------
# Test individual acquisition functions
# ---------------------------------------------------------------------------


class TestPredictionVariance:
    def test_shape(self):
        model = MockUQModel()
        X = np.random.rand(20, 2)
        scores = PredictionVariance().score(X, model)
        assert scores.shape == (20,)

    def test_higher_std_higher_score(self):
        model = MockUQModel(
            std_func=lambda X: np.abs(X[:, 0])  # std proportional to x0
        )
        X = np.array([[0.1, 0], [0.9, 0]])
        scores = PredictionVariance().score(X, model)
        assert scores[1] > scores[0]


class TestUCB:
    def test_shape(self):
        model = MockUQModel()
        X = np.random.rand(15, 2)
        scores = UCB(kappa=2.0).score(X, model)
        assert scores.shape == (15,)

    def test_kappa_effect(self):
        model = MockUQModel()
        X = np.random.rand(10, 2)
        s1 = UCB(kappa=1.0).score(X, model)
        s2 = UCB(kappa=5.0).score(X, model)
        # Higher kappa -> higher scores (since std > 0)
        assert np.all(s2 >= s1)

    def test_formula(self):
        model = MockUQModel(
            mu_func=lambda X: np.ones(len(X)) * 3.0,
            std_func=lambda X: np.ones(len(X)) * 0.5,
        )
        X = np.array([[0, 0]])
        score = UCB(kappa=2.0).score(X, model)
        np.testing.assert_allclose(score, [4.0])  # 3 + 2*0.5


class TestLCB:
    def test_shape(self):
        model = MockUQModel()
        X = np.random.rand(15, 2)
        scores = LCB(kappa=2.0).score(X, model)
        assert scores.shape == (15,)

    def test_formula(self):
        model = MockUQModel(
            mu_func=lambda X: np.ones(len(X)) * 3.0,
            std_func=lambda X: np.ones(len(X)) * 0.5,
        )
        X = np.array([[0, 0]])
        score = LCB(kappa=2.0).score(X, model)
        np.testing.assert_allclose(score, [-2.0])  # -3 + 2*0.5


class TestExpectedImprovement:
    def test_shape(self):
        model = MockUQModel()
        ei = ExpectedImprovement(minimize=True)
        ei._y_best = 0.5
        X = np.random.rand(20, 2)
        scores = ei.score(X, model)
        assert scores.shape == (20,)

    def test_nonnegative(self):
        model = MockUQModel()
        ei = ExpectedImprovement(minimize=True)
        ei._y_best = 0.5
        X = np.random.rand(50, 2)
        scores = ei.score(X, model)
        assert np.all(scores >= -1e-10)

    def test_fallback_to_exploration(self):
        """Without y_best, should return std."""
        model = MockUQModel(std_func=lambda X: np.ones(len(X)) * 0.7)
        ei = ExpectedImprovement()
        X = np.random.rand(5, 2)
        scores = ei.score(X, model)
        np.testing.assert_allclose(scores, 0.7)

    def test_explicit_y_best_overrides(self):
        model = MockUQModel(
            mu_func=lambda X: np.zeros(len(X)),
            std_func=lambda X: np.ones(len(X)),
        )
        ei = ExpectedImprovement(minimize=True, y_best=10.0)
        ei._y_best = 0.0  # should be ignored
        X = np.array([[0, 0]])
        score = ei.score(X, model)
        # With y_best=10, huge improvement expected
        assert score[0] > 5


class TestProbabilityOfImprovement:
    def test_shape(self):
        model = MockUQModel()
        pi = ProbabilityOfImprovement(minimize=True)
        pi._y_best = 0.5
        X = np.random.rand(20, 2)
        scores = pi.score(X, model)
        assert scores.shape == (20,)

    def test_bounded_0_1(self):
        model = MockUQModel()
        pi = ProbabilityOfImprovement(minimize=True)
        pi._y_best = 0.5
        X = np.random.rand(50, 2)
        scores = pi.score(X, model)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)


class TestModelMin:
    def test_prefers_low_mu(self):
        model = MockUQModel(mu_func=lambda X: X[:, 0])
        X = np.array([[0.1, 0], [0.9, 0]])
        scores = ModelMin().score(X, model)
        assert scores[0] > scores[1]  # lower mu -> higher score


class TestModelMax:
    def test_prefers_high_mu(self):
        model = MockUQModel(mu_func=lambda X: X[:, 0])
        X = np.array([[0.1, 0], [0.9, 0]])
        scores = ModelMax().score(X, model)
        assert scores[1] > scores[0]


class TestEnsembleDisagreement:
    def test_shape(self):
        model = MockEnsembleModel()
        X = np.random.rand(10, 2)
        scores = EnsembleDisagreement().score(X, model)
        assert scores.shape == (10,)

    def test_requires_predict_ensemble(self):
        model = MockUQModel()
        with pytest.raises(AttributeError, match="predict_ensemble"):
            EnsembleDisagreement().score(np.random.rand(5, 2), model)


class TestThompsonSampling:
    def test_shape(self):
        model = MockUQModel()
        X = np.random.rand(15, 2)
        scores = ThompsonSampling(random_state=42).score(X, model)
        assert scores.shape == (15,)

    def test_reproducibility(self):
        model = MockUQModel()
        X = np.random.rand(15, 2)
        s1 = ThompsonSampling(random_state=42).score(X, model)
        s2 = ThompsonSampling(random_state=42).score(X, model)
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Test Composite
# ---------------------------------------------------------------------------


class TestComposite:
    def test_add_two(self):
        c = UCB() + PredictionVariance()
        assert isinstance(c, Composite)
        assert len(c.functions) == 2

    def test_scalar_mul(self):
        c = 0.5 * UCB()
        assert isinstance(c, Composite)
        assert c.weights == [0.5]

    def test_weighted_sum(self):
        c = 0.7 * UCB(kappa=2) + 0.3 * PredictionVariance()
        assert isinstance(c, Composite)
        assert len(c.functions) == 2
        assert c.weights == pytest.approx([0.7, 0.3])

    def test_score_shape(self):
        model = MockUQModel()
        c = 0.5 * UCB() + 0.5 * PredictionVariance()
        X = np.random.rand(20, 2)
        scores = c.score(X, model)
        assert scores.shape == (20,)

    def test_normalization(self):
        """Composite output should be bounded by sum of weights."""
        model = MockUQModel()
        c = 0.5 * UCB() + 0.5 * PredictionVariance()
        X = np.random.rand(50, 2)
        scores = c.score(X, model)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1.0 + 1e-10)

    def test_name(self):
        c = 0.7 * UCB() + 0.3 * PredictionVariance()
        assert "UCB" in c.name
        assert "PredictionVariance" in c.name

    def test_composite_add_composite(self):
        c1 = 0.5 * UCB()
        c2 = 0.3 * LCB()
        c3 = c1 + c2
        assert isinstance(c3, Composite)
        assert len(c3.functions) == 2


# ---------------------------------------------------------------------------
# Test ActiveLearner
# ---------------------------------------------------------------------------


class TestActiveLearner:
    def setup_method(self):
        self.model = MockUQModel()
        self.bounds = [(0, 1), (0, 1)]
        self.acq = UCB(kappa=2.0)

    def test_suggest_shape(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        result = learner.suggest(n_points=5)
        assert result.points.shape == (5, 2)
        assert result.scores.shape == (5,)

    def test_suggest_greedy(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        result = learner.suggest(n_points=3, batch_strategy="greedy")
        assert result.metadata["batch_strategy"] == "greedy"
        assert len(result.points) == 3

    def test_suggest_penalized(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        result = learner.suggest(n_points=3, batch_strategy="penalized")
        assert result.metadata["batch_strategy"] == "penalized"
        assert len(result.points) == 3

    def test_suggest_thompson(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        result = learner.suggest(n_points=3, batch_strategy="thompson")
        assert result.metadata["batch_strategy"] == "thompson"
        assert len(result.points) == 3

    def test_suggest_custom_candidates(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        custom = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        result = learner.suggest(n_points=2, candidates=custom)
        assert result.points.shape == (2, 2)

    def test_update_grows_data(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        learner.update(np.array([[0.5, 0.5]]), np.array([1.0]), refit=False)
        assert learner.n_observations == 1
        assert learner.iteration == 1

        learner.update(np.array([[0.3, 0.7], [0.1, 0.2]]), np.array([2.0, 3.0]), refit=False)
        assert learner.n_observations == 3
        assert learner.iteration == 2

    def test_update_refit_calls_fit(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        learner.update(np.array([[0.5, 0.5]]), np.array([1.0]), refit=True)
        assert self.model._fit_count == 1

    def test_update_no_refit(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        learner.update(np.array([[0.5, 0.5]]), np.array([1.0]), refit=False)
        assert self.model._fit_count == 0

    def test_best_y_tracking(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        assert learner.best_y is None
        learner.update(np.array([[0.5, 0.5]]), np.array([3.0]), refit=False)
        assert learner.best_y == 3.0
        learner.update(np.array([[0.1, 0.1]]), np.array([1.0]), refit=False)
        assert learner.best_y == 1.0

    def test_best_X_tracking(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        learner.update(
            np.array([[0.5, 0.5], [0.1, 0.1]]),
            np.array([3.0, 1.0]),
            refit=False,
        )
        np.testing.assert_array_equal(learner.best_X, [0.1, 0.1])

    def test_reproducibility(self):
        r1 = ActiveLearner(self.model, self.bounds, self.acq, random_state=42).suggest(n_points=5)
        r2 = ActiveLearner(self.model, self.bounds, self.acq, random_state=42).suggest(n_points=5)
        np.testing.assert_array_equal(r1.points, r2.points)

    def test_get_set_params(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, n_candidates=500)
        params = learner.get_params()
        assert params["n_candidates"] == 500
        learner.set_params(n_candidates=2000)
        assert learner.n_candidates == 2000

    def test_set_params_invalid(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq)
        with pytest.raises(ValueError, match="Invalid parameter"):
            learner.set_params(nonexistent=42)

    def test_candidate_methods(self):
        for method in ["lhs", "sobol", "halton", "random"]:
            learner = ActiveLearner(
                self.model,
                self.bounds,
                self.acq,
                candidate_method=method,
                random_state=42,
            )
            result = learner.suggest(n_points=3)
            assert result.points.shape == (3, 2)

    def test_invalid_batch_strategy(self):
        learner = ActiveLearner(self.model, self.bounds, self.acq, random_state=42)
        with pytest.raises(ValueError, match="batch_strategy"):
            learner.suggest(batch_strategy="invalid")

    def test_invalid_candidate_method(self):
        learner = ActiveLearner(
            self.model,
            self.bounds,
            self.acq,
            candidate_method="invalid",
            random_state=42,
        )
        with pytest.raises(ValueError, match="candidate_method"):
            learner.suggest()


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_loop_with_linear_regression_uq(self):
        """Full active learning loop with LinearRegressionUQ."""
        from pycse.sklearn.lr_uq import LinearRegressionUQ

        rng = np.random.RandomState(42)

        # Oracle: y = 2*x0 + 3*x1 + noise
        def oracle(X):
            return 2 * X[:, 0] + 3 * X[:, 1] + rng.normal(0, 0.1, len(X))

        # Initial data
        X_init = rng.uniform(0, 1, size=(10, 2))
        y_init = oracle(X_init)

        model = LinearRegressionUQ()
        model.fit(X_init, y_init)

        learner = ActiveLearner(
            model=model,
            bounds=[(0, 1), (0, 1)],
            acquisition=PredictionVariance(),
            n_candidates=200,
            random_state=42,
        )
        learner.X_train = X_init.copy()
        learner.y_train = y_init.copy()

        # Run 3 iterations
        for _ in range(3):
            result = learner.suggest(n_points=3)
            assert result.points.shape == (3, 2)
            y_new = oracle(result.points)
            learner.update(result.points, y_new, refit=True)

        assert learner.n_observations == 10 + 9  # 10 initial + 3*3
        assert learner.iteration == 3

    def test_ei_with_mock(self):
        """EI gets y_best set by ActiveLearner."""
        model = MockUQModel(
            mu_func=lambda X: np.sum(X, axis=1),
            std_func=lambda X: np.ones(len(X)) * 0.3,
        )
        ei = ExpectedImprovement(minimize=True)
        learner = ActiveLearner(
            model=model,
            bounds=[(0, 1), (0, 1)],
            acquisition=ei,
            random_state=42,
        )
        learner.update(np.array([[0.5, 0.5]]), np.array([1.0]), refit=False)
        result = learner.suggest(n_points=3)
        # EI should be non-negative
        assert np.all(result.scores >= -1e-10)
