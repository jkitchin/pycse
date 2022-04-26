"""Test module for PYCSE.py."""

import numpy as np
from pycse.PYCSE import polyfit, regress, nlinfit, Rsquared, ivp


def test_polyfit():
    """Test on fitting a line.

    I don't know if there are good ways to test that bint, se are correct.
    """
    x = np.array([0, 1])
    y = np.array([0, 1])

    b, bint, se = polyfit(x, y, 1)

    assert np.isclose(b[0], 1.0)
    assert np.isclose(b[1], 0.0)


def test_regress_defaults():
    x = np.array([0, 1])
    y = np.array([0, 1])

    X = np.column_stack([x, x**0])

    b, bint, se = regress(X, y)

    assert np.isclose(b[0], 1.0)
    assert np.isclose(b[1], 0.0)
    assert bint is None
    assert se is None


def test_regress():
    x = np.array([0, 1])
    y = np.array([0, 1])

    X = np.column_stack([x, x**0])

    b, bint, se = regress(X, y, 0.05)

    assert np.isclose(b[0], 1.0)
    assert np.isclose(b[1], 0.0)
    assert bint is not None
    assert se is not None


def test_nlinfit_defaults():
    x = np.array([0, 1])
    y = np.array([0, 1])

    def f(x, m, b):
        return m * x + b

    b, bint, se = nlinfit(f, x, y, [0.5, 0.5])
    assert np.isclose(b[0], 1.0)
    assert np.isclose(b[1], 0.0)
    assert bint is not None
    assert se is not None


def test_rsquared():
    x = np.array([0, 1])
    y = np.array([0, 1])
    assert np.isclose(1.0, Rsquared(x, y))


def test_ivp():
    def ode(x, y):
        return y

    # You need good tolerance here.
    sol = ivp(ode, np.array([0, 1]), [1], rtol=1e-8, atol=1e-8)

    assert np.isclose(np.exp(1), sol.y[0][-1])
