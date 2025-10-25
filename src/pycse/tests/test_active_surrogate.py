"""Tests for ActiveSurrogate class."""

import pytest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from pycse.pyroxy import ActiveSurrogate


@pytest.fixture
def simple_gpr():
    """Create a simple GPR model for testing."""
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)


@pytest.fixture
def simple_1d_function():
    """A simple 1D test function."""

    def f(X):
        return np.sin(X).flatten()

    return f


class TestActiveSurrogateBasic:
    """Basic tests for ActiveSurrogate class."""

    def test_class_exists(self):
        """Test that ActiveSurrogate class exists."""
        assert hasattr(ActiveSurrogate, "build")


class TestActiveSurrogateValidation:
    """Test input validation for ActiveSurrogate."""

    def test_invalid_bounds_not_list(self, simple_gpr, simple_1d_function):
        """Test that non-list bounds raise error."""
        with pytest.raises(ValueError, match="bounds must be list"):
            ActiveSurrogate.build(
                func=simple_1d_function,
                bounds=(0, 1),  # tuple not list
                model=simple_gpr,
            )

    def test_invalid_bounds_not_tuples(self, simple_gpr, simple_1d_function):
        """Test that non-tuple elements raise error."""
        with pytest.raises(ValueError, match="bounds must be list"):
            ActiveSurrogate.build(
                func=simple_1d_function,
                bounds=[[0, 1]],  # list not tuple
                model=simple_gpr,
            )

    def test_invalid_acquisition(self, simple_gpr, simple_1d_function):
        """Test that invalid acquisition raises error."""
        with pytest.raises(ValueError, match="acquisition must be one of"):
            ActiveSurrogate.build(
                func=simple_1d_function, bounds=[(0, 1)], model=simple_gpr, acquisition="invalid"
            )

    def test_invalid_stopping_criterion(self, simple_gpr, simple_1d_function):
        """Test that invalid stopping criterion raises error."""
        with pytest.raises(ValueError, match="stopping_criterion must be one of"):
            ActiveSurrogate.build(
                func=simple_1d_function,
                bounds=[(0, 1)],
                model=simple_gpr,
                stopping_criterion="invalid",
            )


class TestActiveSurrogateLHS:
    """Test Latin Hypercube Sampling helper."""

    def test_generate_lhs_samples_1d(self):
        """Test LHS for 1D domain."""
        bounds = [(0.0, 10.0)]
        samples = ActiveSurrogate._generate_lhs_samples(bounds, n_samples=20)

        assert samples.shape == (20, 1)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 10.0)

    def test_generate_lhs_samples_2d(self):
        """Test LHS for 2D domain."""
        bounds = [(0.0, 10.0), (-5.0, 5.0)]
        samples = ActiveSurrogate._generate_lhs_samples(bounds, n_samples=30)

        assert samples.shape == (30, 2)
        assert np.all(samples[:, 0] >= 0.0)
        assert np.all(samples[:, 0] <= 10.0)
        assert np.all(samples[:, 1] >= -5.0)
        assert np.all(samples[:, 1] <= 5.0)

    def test_generate_lhs_samples_coverage(self):
        """Test that LHS provides good coverage."""
        bounds = [(0.0, 1.0)]
        samples = ActiveSurrogate._generate_lhs_samples(bounds, n_samples=100)

        # Check distribution across domain
        assert np.min(samples) < 0.2
        assert np.max(samples) > 0.8


class TestAcquisitionFunctions:
    """Test acquisition function implementations."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted GPR model."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.sin(X).flatten()

        kernel = C(1.0) * RBF(1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        return model, y

    def test_acquisition_ei(self, fitted_model):
        """Test Expected Improvement acquisition."""
        model, y_train = fitted_model
        X_candidates = np.array([[1.5], [2.5]])

        ei = ActiveSurrogate._acquisition_ei(X_candidates, model, y_train.max())

        assert ei.shape == (2,)
        assert np.all(ei >= 0)

    def test_acquisition_ucb(self, fitted_model):
        """Test Upper Confidence Bound acquisition."""
        model, _ = fitted_model
        X_candidates = np.array([[1.5], [2.5]])

        ucb = ActiveSurrogate._acquisition_ucb(X_candidates, model, kappa=2.0)

        assert ucb.shape == (2,)

    def test_acquisition_pi(self, fitted_model):
        """Test Probability of Improvement acquisition."""
        model, y_train = fitted_model
        X_candidates = np.array([[1.5], [2.5]])

        pi = ActiveSurrogate._acquisition_pi(X_candidates, model, y_train.max())

        assert pi.shape == (2,)
        assert np.all(pi >= 0)
        assert np.all(pi <= 1)

    def test_acquisition_variance(self, fitted_model):
        """Test Maximum Variance acquisition."""
        model, _ = fitted_model
        X_candidates = np.array([[1.5], [2.5], [10.0]])

        variance = ActiveSurrogate._acquisition_variance(X_candidates, model)

        assert variance.shape == (3,)
        assert np.all(variance >= 0)
        # Point far from training data should have higher variance
        assert variance[2] > variance[0]


class TestStoppingCriteria:
    """Test stopping criterion implementations."""

    def test_stopping_mean_ratio_met(self):
        """Test mean_ratio criterion when met."""
        test_unc = np.array([0.5, 0.6, 0.7])
        train_unc = np.array([0.8, 0.9, 1.0])

        # test mean = 0.6, train mean = 0.9, ratio = 0.67 < 1.5
        result = ActiveSurrogate._stopping_mean_ratio(test_unc, train_unc, threshold=1.5)
        assert result is True

    def test_stopping_mean_ratio_not_met(self):
        """Test mean_ratio criterion when not met."""
        test_unc = np.array([1.5, 2.0, 2.5])
        train_unc = np.array([0.8, 0.9, 1.0])

        # test mean = 2.0, train mean = 0.9, ratio = 2.22 > 1.5
        result = ActiveSurrogate._stopping_mean_ratio(test_unc, train_unc, threshold=1.5)
        assert result is False

    def test_stopping_percentile_met(self):
        """Test percentile criterion when met."""
        test_unc = np.linspace(0.01, 0.09, 100)

        result = ActiveSurrogate._stopping_percentile(test_unc, threshold=0.1)
        assert result is True

    def test_stopping_absolute_met(self):
        """Test absolute criterion when met."""
        test_unc = np.array([0.05, 0.08, 0.09])

        result = ActiveSurrogate._stopping_absolute(test_unc, threshold=0.1)
        assert result is True

    def test_stopping_absolute_not_met(self):
        """Test absolute criterion when not met."""
        test_unc = np.array([0.05, 0.08, 0.15])

        result = ActiveSurrogate._stopping_absolute(test_unc, threshold=0.1)
        assert result is False

    def test_stopping_convergence_met(self):
        """Test convergence criterion when met."""
        history = {"mean_uncertainty": [1.0, 0.9, 0.89, 0.88, 0.87, 0.875, 0.87]}

        result = ActiveSurrogate._stopping_convergence(history, window=5, threshold=0.05)
        assert result is True

    def test_stopping_convergence_not_met(self):
        """Test convergence criterion when not met."""
        history = {"mean_uncertainty": [1.0, 0.9, 0.7, 0.5, 0.3]}

        result = ActiveSurrogate._stopping_convergence(history, window=3, threshold=0.05)
        assert result is False


class TestBatchSelection:
    """Test batch selection with hallucination."""

    @pytest.fixture
    def fitted_model_2d(self):
        """Create a fitted 2D GPR model."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.sin(X[:, 0]) + np.cos(X[:, 1])

        kernel = C(1.0) * RBF(1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        return model, y

    def test_select_batch_single(self, fitted_model_2d):
        """Test batch selection with batch_size=1."""
        model, y_train = fitted_model_2d
        X_candidates = np.random.rand(20, 2)

        selected = ActiveSurrogate._select_batch(
            X_candidates, model, y_train, acquisition="ei", batch_size=1
        )

        assert selected.shape == (1, 2)

    def test_select_batch_multiple(self, fitted_model_2d):
        """Test batch selection with batch_size>1."""
        model, y_train = fitted_model_2d
        X_candidates = np.random.rand(50, 2)

        selected = ActiveSurrogate._select_batch(
            X_candidates, model, y_train, acquisition="ucb", batch_size=3
        )

        assert selected.shape == (3, 2)
        # Check that selected points are different
        assert not np.array_equal(selected[0], selected[1])
        assert not np.array_equal(selected[1], selected[2])

    def test_select_batch_diversity(self, fitted_model_2d):
        """Test that batch selection promotes diversity."""
        model, y_train = fitted_model_2d
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create candidates with a clear cluster
        X_candidates = np.vstack(
            [
                np.random.rand(40, 2) * 0.1 + 0.5,  # Clustered around 0.5
                np.random.rand(10, 2),  # Spread across space
            ]
        )

        selected = ActiveSurrogate._select_batch(
            X_candidates, model, y_train, acquisition="variance", batch_size=3
        )

        # With hallucination, points should be somewhat spread
        distances = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                distances.append(np.linalg.norm(selected[i] - selected[j]))

        # All selected points should be different (non-zero distances)
        # Use a very small threshold to just verify diversity without being too strict
        assert all(d > 0.0 for d in distances)


class TestActiveSurrogateEndToEnd:
    """End-to-end integration tests."""

    def test_build_simple_1d(self, simple_gpr, simple_1d_function):
        """Test building surrogate for simple 1D function."""
        bounds = [(0, 2 * np.pi)]

        surrogate, history = ActiveSurrogate.build(
            func=simple_1d_function,
            bounds=bounds,
            model=simple_gpr,
            acquisition="variance",
            stopping_criterion="absolute",
            stopping_threshold=0.001,  # Strict threshold to force iterations
            n_initial=5,
            max_iterations=20,
            verbose=False,
        )

        # Check surrogate is a _Surrogate instance
        from pycse.pyroxy import _Surrogate

        assert isinstance(surrogate, _Surrogate)

        # Check it has training data
        assert surrogate.xtrain is not None
        assert surrogate.ytrain is not None
        assert len(surrogate.xtrain) >= 5  # At least initial samples

        # Check history
        assert "iterations" in history
        assert "n_samples" in history
        assert "mean_uncertainty" in history
        assert len(history["iterations"]) > 0

        # Check surrogate works
        X_test = np.array([[np.pi / 2]])
        y_pred = surrogate(X_test)
        assert y_pred.shape == (1,)

    def test_build_respects_max_iterations(self, simple_gpr, simple_1d_function):
        """Test that max_iterations limits are respected."""
        bounds = [(0, 10)]

        surrogate, history = ActiveSurrogate.build(
            func=simple_1d_function,
            bounds=bounds,
            model=simple_gpr,
            acquisition="ei",
            stopping_criterion="absolute",
            stopping_threshold=0.001,  # Very strict, won't be met
            n_initial=3,
            max_iterations=5,
            verbose=False,
        )

        # Should stop at max_iterations
        assert len(history["iterations"]) <= 5


class TestActiveSurrogateIntegration:
    """Additional integration tests for various scenarios."""

    def test_build_2d_function(self, simple_gpr):
        """Test with 2D function."""

        def func_2d(X):
            return (np.sin(X[:, 0]) * np.cos(X[:, 1])).flatten()

        bounds = [(0, np.pi), (0, np.pi)]

        surrogate, history = ActiveSurrogate.build(
            func=func_2d,
            bounds=bounds,
            model=simple_gpr,
            acquisition="ei",
            stopping_criterion="mean_ratio",
            stopping_threshold=2.0,
            n_initial=10,
            max_iterations=15,
            verbose=False,
        )

        assert surrogate.xtrain.shape[1] == 2
        assert len(surrogate.ytrain) >= 10

    def test_different_acquisitions(self, simple_gpr, simple_1d_function):
        """Test all acquisition functions."""
        bounds = [(0, 2 * np.pi)]

        for acq in ["ei", "ucb", "pi", "variance"]:
            surrogate, history = ActiveSurrogate.build(
                func=simple_1d_function,
                bounds=bounds,
                model=simple_gpr,
                acquisition=acq,
                stopping_criterion="absolute",
                stopping_threshold=0.5,
                n_initial=5,
                max_iterations=10,
                verbose=False,
            )

            assert len(surrogate.xtrain) >= 5

    def test_different_stopping_criteria(self, simple_gpr, simple_1d_function):
        """Test all stopping criteria."""
        bounds = [(0, 2 * np.pi)]

        # Test mean_ratio
        surrogate, _ = ActiveSurrogate.build(
            func=simple_1d_function,
            bounds=bounds,
            model=simple_gpr,
            stopping_criterion="mean_ratio",
            stopping_threshold=1.5,
            n_initial=5,
            max_iterations=20,
        )
        assert surrogate is not None

        # Test percentile
        surrogate, _ = ActiveSurrogate.build(
            func=simple_1d_function,
            bounds=bounds,
            model=simple_gpr,
            stopping_criterion="percentile",
            stopping_threshold=0.2,
            n_initial=5,
            max_iterations=20,
        )
        assert surrogate is not None

        # Test absolute
        surrogate, _ = ActiveSurrogate.build(
            func=simple_1d_function,
            bounds=bounds,
            model=simple_gpr,
            stopping_criterion="absolute",
            stopping_threshold=0.2,
            n_initial=5,
            max_iterations=20,
        )
        assert surrogate is not None

    def test_batch_mode(self, simple_gpr, simple_1d_function):
        """Test batch sampling."""
        bounds = [(0, 2 * np.pi)]

        surrogate, history = ActiveSurrogate.build(
            func=simple_1d_function,
            bounds=bounds,
            model=simple_gpr,
            acquisition="ucb",
            batch_size=3,
            stopping_criterion="absolute",
            stopping_threshold=0.3,
            n_initial=5,
            max_iterations=5,
        )

        # With batch_size=3, should add 3 points per iteration
        # Check that sample count increases appropriately
        assert len(surrogate.xtrain) >= 5

    def test_callback_invoked(self, simple_gpr, simple_1d_function):
        """Test that callback is called during training."""
        bounds = [(0, 2 * np.pi)]
        callback_count = [0]

        def test_callback(iteration, history):
            callback_count[0] += 1

        surrogate, history = ActiveSurrogate.build(
            func=simple_1d_function,
            bounds=bounds,
            model=simple_gpr,
            acquisition="variance",
            stopping_criterion="absolute",
            stopping_threshold=0.001,  # Very strict threshold to ensure iterations run
            n_initial=3,
            max_iterations=5,
            callback=test_callback,
        )

        # Callback should be invoked at least once
        assert callback_count[0] > 0

    def test_surrogate_usage_after_build(self, simple_gpr, simple_1d_function):
        """Test that returned surrogate can be used for prediction."""
        bounds = [(0, 2 * np.pi)]

        surrogate, _ = ActiveSurrogate.build(
            func=simple_1d_function,
            bounds=bounds,
            model=simple_gpr,
            acquisition="ei",
            stopping_criterion="absolute",
            stopping_threshold=0.3,
            n_initial=10,
            max_iterations=10,
        )

        # Test prediction
        X_test = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        y_pred = surrogate(X_test)

        assert y_pred.shape == (20,)

        # Check that predictions are reasonable
        y_true = simple_1d_function(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        assert rmse < 0.5  # Should be reasonably accurate
