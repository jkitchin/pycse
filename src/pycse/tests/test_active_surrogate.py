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
