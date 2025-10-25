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
