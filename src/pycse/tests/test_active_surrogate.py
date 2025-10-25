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
