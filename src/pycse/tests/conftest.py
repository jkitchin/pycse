"""Shared pytest fixtures and utilities for pycse tests.

This module provides common fixtures and helper functions used across
multiple test modules to reduce code duplication and improve test
maintainability.
"""

import os

# Force JAX to use CPU backend to avoid Metal GPU issues on macOS
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pytest


# Sample data fixtures for regression tests
@pytest.fixture
def linear_data():
    """Simple linear relationship: y = 2x + 1."""
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x + 1
    return x, y


@pytest.fixture
def linear_data_with_noise():
    """Linear relationship with added noise."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
    return x, y


@pytest.fixture
def quadratic_data():
    """Quadratic relationship: y = x^2 + 2x + 1."""
    x = np.array([0, 1, 2, 3, 4])
    y = x**2 + 2 * x + 1
    return x, y


@pytest.fixture
def exponential_data():
    """Exponential relationship: y = exp(x)."""
    x = np.linspace(0, 2, 10)
    y = np.exp(x)
    return x, y


@pytest.fixture
def simple_ode_solution():
    """Solution to dy/dx = y with y(0) = 1: y = exp(x)."""
    x = np.linspace(0, 1, 10)
    y = np.exp(x)
    return x, y


# Helper functions for numerical comparisons
def assert_allclose_with_tol(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert arrays are close within tolerance.

    Parameters
    ----------
    actual : array_like
        Actual values from computation
    expected : array_like
        Expected values
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    """
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_shape(arr, expected_shape):
    """Assert array has expected shape.

    Parameters
    ----------
    arr : array_like
        Array to check
    expected_shape : tuple
        Expected shape
    """
    assert arr.shape == expected_shape, f"Expected shape {expected_shape}, got {arr.shape}"


def assert_confidence_interval_valid(ci, point_estimate):
    """Assert confidence interval is valid.

    A valid confidence interval should:
    1. Have lower bound < upper bound
    2. Contain the point estimate

    Parameters
    ----------
    ci : array_like, shape (2,)
        Confidence interval [lower, upper]
    point_estimate : float
        Point estimate that should be within CI
    """
    lower, upper = ci
    assert lower < upper, f"CI lower ({lower}) >= upper ({upper})"
    assert lower <= point_estimate <= upper, (
        f"Point estimate {point_estimate} not in CI [{lower}, {upper}]"
    )


def assert_positive(arr):
    """Assert all elements in array are positive.

    Parameters
    ----------
    arr : array_like
        Array to check
    """
    assert np.all(arr > 0), f"Expected all positive values, got min={np.min(arr)}"


def assert_standard_errors_positive(se):
    """Assert standard errors are positive.

    Standard errors must be strictly positive.

    Parameters
    ----------
    se : array_like
        Standard errors
    """
    assert_positive(se)


# Autouse fixture to speed up slow tests
@pytest.fixture(autouse=True)
def fast_training_defaults(monkeypatch):
    """Override model defaults to use minimal iterations for tests.

    This fixture automatically reduces training iterations to 10 for all models
    during testing to speed up CI. Models still use reasonable defaults (50)
    for actual usage, but tests only need to verify functionality, not convergence.
    """
    # Import all the model classes
    try:
        from pycse.sklearn.jax_icnn import JAXICNNRegressor

        orig_icnn_init = JAXICNNRegressor.__init__

        def fast_icnn_init(self, *args, epochs=10, **kwargs):
            orig_icnn_init(self, *args, epochs=epochs, **kwargs)

        monkeypatch.setattr(JAXICNNRegressor, "__init__", fast_icnn_init)
    except ImportError:
        pass

    try:
        from pycse.sklearn.jax_monotonic import JAXMonotonicRegressor

        orig_mono_init = JAXMonotonicRegressor.__init__

        def fast_mono_init(self, *args, epochs=10, **kwargs):
            orig_mono_init(self, *args, epochs=epochs, **kwargs)

        monkeypatch.setattr(JAXMonotonicRegressor, "__init__", fast_mono_init)
    except ImportError:
        pass

    try:
        from pycse.sklearn.jax_periodic import JAXPeriodicRegressor

        orig_periodic_init = JAXPeriodicRegressor.__init__

        def fast_periodic_init(self, *args, epochs=10, **kwargs):
            orig_periodic_init(self, *args, epochs=epochs, **kwargs)

        monkeypatch.setattr(JAXPeriodicRegressor, "__init__", fast_periodic_init)
    except ImportError:
        pass

    try:
        from pycse.sklearn.dpose import DPOSE

        orig_dpose_fit = DPOSE.fit

        def fast_dpose_fit(self, X, y, **kwargs):
            kwargs.setdefault("maxiter", 10)
            return orig_dpose_fit(self, X, y, **kwargs)

        monkeypatch.setattr(DPOSE, "fit", fast_dpose_fit)
    except ImportError:
        pass

    try:
        from pycse.sklearn.kfoldnn import KfoldNN

        orig_kfold_fit = KfoldNN.fit

        def fast_kfold_fit(self, X, y, **kwargs):
            kwargs.setdefault("maxiter", 10)
            return orig_kfold_fit(self, X, y, **kwargs)

        monkeypatch.setattr(KfoldNN, "fit", fast_kfold_fit)
    except ImportError:
        pass

    try:
        from pycse.sklearn.kan import KANRegressor

        orig_kan_fit = KANRegressor.fit

        def fast_kan_fit(self, X, y, **kwargs):
            kwargs.setdefault("maxiter", 10)
            return orig_kan_fit(self, X, y, **kwargs)

        monkeypatch.setattr(KANRegressor, "fit", fast_kan_fit)
    except ImportError:
        pass

    try:
        from pycse.sklearn.kan_llpr import KANLLPRRegressor

        orig_kanllpr_fit = KANLLPRRegressor.fit

        def fast_kanllpr_fit(self, X, y, **kwargs):
            kwargs.setdefault("maxiter", 10)
            return orig_kanllpr_fit(self, X, y, **kwargs)

        monkeypatch.setattr(KANLLPRRegressor, "fit", fast_kanllpr_fit)
    except ImportError:
        pass


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests (critical path)"
    )
    config.addinivalue_line("markers", "sklearn: marks tests for sklearn module")
