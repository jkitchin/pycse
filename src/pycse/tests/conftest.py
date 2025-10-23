"""Shared pytest fixtures and utilities for pycse tests.

This module provides common fixtures and helper functions used across
multiple test modules to reduce code duplication and improve test
maintainability.
"""

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
