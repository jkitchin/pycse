"""Regression tests for interval calculations and API fixes in PYCSE.py.

The prediction intervals are checked against exact textbook / delta-method
formulas computed independently here.
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.stats import t

from pycse.PYCSE import ivp, nlpredict, polyfit, polyval, predict, regress


def _line_data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + rng.normal(0, 0.5, n)
    return x, y


def _textbook_pi(X, y, XX, alpha=0.05):
    """Exact OLS prediction interval: yhat +- t * s * sqrt(1 + x0 (X'X)^-1 x0')."""
    n, p = X.shape
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = np.sum((y - X @ b) ** 2) / (n - p)
    lev = np.einsum("ij,jk,ik->i", XX, np.linalg.inv(X.T @ X), XX)
    se = np.sqrt(s2 * (1 + lev))
    tval = t.ppf(1 - alpha / 2, n - p)
    yy = XX @ b
    return yy, np.column_stack([yy - tval * se, yy + tval * se]), se


def test_predict_matches_textbook_extrapolated():
    """No always-on regularization: exact interval, even far outside the data."""
    x, y = _line_data()
    X = np.column_stack([x, x**0])
    pars, _, _ = regress(X, y)
    XX = np.array([[50.0, 1.0], [5.0, 1.0], [-20.0, 1.0]])

    yy, yint, se = predict(X, y, pars, XX)
    yy0, yint0, se0 = _textbook_pi(X, y, XX)

    # Before the fix the ratio se / se0 was 0.82 at x = 50.
    np.testing.assert_allclose(se, se0, rtol=1e-8)
    np.testing.assert_allclose(yint, yint0, rtol=1e-8)
    np.testing.assert_allclose(yy, yy0, rtol=1e-8)


def test_polyval_matches_textbook():
    x, y = _line_data()
    p, _, _ = polyfit(x, y, 2)
    xnew = np.array([-5.0, 3.3, 50.0])

    yy, yint, se = polyval(p, xnew, x, y)
    yy0, yint0, se0 = _textbook_pi(np.vander(x, 3), y, np.vander(xnew, 3))

    np.testing.assert_allclose(se, se0, rtol=1e-6)
    np.testing.assert_allclose(yint, yint0, rtol=1e-6)


def test_predict_multi_output_matches_textbook():
    x, y1 = _line_data(seed=1)
    _, y2 = _line_data(seed=2)
    Y = np.column_stack([y1, 3 * y2])
    X = np.column_stack([x, x**0])
    pars, _, _ = regress(X, Y)
    XX = np.array([[50.0, 1.0], [2.0, 1.0]])

    yy, yint, se = predict(X, Y, pars, XX)
    for k in range(2):
        _, _, se0 = _textbook_pi(X, Y[:, k], XX)
        np.testing.assert_allclose(se[:, k], se0, rtol=1e-8)


def test_predict_singular_still_regularized():
    """Collinear columns: X'X is singular, regularization must still kick in."""
    x, y = _line_data()
    X = np.column_stack([x, x, x**0])
    pars = np.linalg.lstsq(X, y, rcond=None)[0]
    yy, yint, se = predict(X, y, pars, np.array([[5.0, 5.0, 1.0]]))
    assert np.all(np.isfinite(se)) and np.all(np.isfinite(yint))


def _exp_data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 3, 25)
    y = 2 * np.exp(-0.7 * x) + rng.normal(0, 0.05, x.size)
    return x, y


def _exp_model(x, a, b):
    return a * np.exp(-b * x)


def test_nlpredict_matches_delta_method():
    """nlpredict intervals were ~sqrt(2) too narrow (sigma^2 was halved)."""
    x, y = _exp_data()
    popt, pcov = curve_fit(_exp_model, x, y, p0=[1, 1])
    xnew = np.array([0.5, 1.5, 4.0])
    n, p = len(x), len(popt)

    a, b = popt
    s2 = np.sum((y - _exp_model(x, *popt)) ** 2) / (n - p)
    J = np.column_stack([np.exp(-b * x), -a * x * np.exp(-b * x)])
    cov = s2 * np.linalg.inv(J.T @ J)
    np.testing.assert_allclose(cov, pcov, rtol=1e-3)

    Jn = np.column_stack([np.exp(-b * xnew), -a * xnew * np.exp(-b * xnew)])
    se_mean = np.sqrt(np.diag(Jn @ cov @ Jn.T))
    se_mean_pcov = np.sqrt(np.diag(Jn @ pcov @ Jn.T))
    tval = t.ppf(0.975, n - p)
    half = tval * np.sqrt(se_mean**2 + s2)

    yp, yint, se = nlpredict(x, y, _exp_model, popt, xnew)

    # nlpredict uses the full Hessian of the loss (not the Gauss-Newton J^T J),
    # so allow a small difference. Before the fix, the ratio was ~0.6-0.7.
    np.testing.assert_allclose(se, se_mean, rtol=2e-2)
    np.testing.assert_allclose(se, se_mean_pcov, rtol=2e-2)
    np.testing.assert_allclose(yint[:, 1] - yp, half, rtol=5e-3)
    np.testing.assert_allclose(yp - yint[:, 0], half, rtol=5e-3)


def test_nlpredict_linear_matches_predict():
    """For a linear model the Hessian is exactly J^T J, so results must match predict."""
    x, y = _line_data()
    popt, _ = curve_fit(lambda x, m, b: m * x + b, x, y)
    xnew = np.array([-3.0, 5.0, 50.0])

    yp, yint, se_mean = nlpredict(x, y, lambda x, m, b: m * x + b, popt, xnew)
    yy0, yint0, se0 = _textbook_pi(np.column_stack([x, x**0]), y, np.column_stack([xnew, xnew**0]))

    np.testing.assert_allclose(yint, yint0, rtol=1e-5)
    s2 = np.sum((y - (popt[0] * x + popt[1])) ** 2) / (len(x) - 2)
    np.testing.assert_allclose(np.sqrt(se_mean**2 + s2), se0, rtol=1e-5)


@pytest.mark.parametrize(
    "loss_factory",
    [
        lambda x, y: lambda *p: np.sum((y - _exp_model(x, *p)) ** 2),
        lambda x, y: lambda *p: np.mean((y - _exp_model(x, *p)) ** 2),
    ],
    ids=["sse", "mse"],
)
def test_nlpredict_custom_loss_scale_invariant(loss_factory):
    """Any loss proportional to SSE gives the same intervals as the default ½SSE."""
    x, y = _exp_data(1)
    popt, _ = curve_fit(_exp_model, x, y, p0=[1, 1])
    xnew = np.array([0.5, 4.0])

    ref = nlpredict(x, y, _exp_model, popt, xnew)
    res = nlpredict(x, y, _exp_model, popt, xnew, loss=loss_factory(x, y))

    np.testing.assert_allclose(res[1], ref[1], rtol=1e-6)
    np.testing.assert_allclose(res[2], ref[2], rtol=1e-6)


def test_regress_alpha_none():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 2.0, 3.9, 6.1])
    X = np.column_stack([x**0, x])
    b, bint, se = regress(X, y, alpha=None)
    assert bint is None and se is None
    np.testing.assert_allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])


def test_ivp_extra_args_passed_to_f():
    tspan = np.linspace(0, 1, 11)
    sol = ivp(lambda t, y, k: -k * y, tspan, [1.0], 2.0, rtol=1e-8, atol=1e-10)
    assert sol.success
    np.testing.assert_allclose(sol.y[0], np.exp(-2.0 * tspan), rtol=1e-6)


def test_ivp_args_keyword_still_works():
    tspan = np.linspace(0, 1, 11)
    sol = ivp(lambda t, y, k: -k * y, tspan, [1.0], args=(2.0,), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sol.y[0], np.exp(-2.0 * tspan), rtol=1e-6)


def test_ivp_args_both_ways_raises():
    with pytest.raises(TypeError):
        ivp(lambda t, y, k: -k * y, [0, 1], [1.0], 2.0, args=(2.0,))


def test_interval_layout_is_consistent():
    """predict, polyval, nlpredict, regress and nlinfit all put (lower, upper) on the last axis."""
    x, y = _line_data()
    X = np.column_stack([x, x**0])
    pars, pint, _ = regress(X, y)
    assert pint.shape == (2, 2)
    xnew = np.array([-3.0, 5.0, 50.0])
    XX = np.column_stack([xnew, xnew**0])

    _, yint_lin, _ = predict(X, y, pars, XX)
    _, yint_nl, _ = nlpredict(x, y, lambda x, m, b: m * x + b, pars, xnew)
    assert yint_lin.shape == yint_nl.shape == (3, 2)
    np.testing.assert_allclose(yint_lin, yint_nl, rtol=1e-5)

    p, _, _ = polyfit(x, y, 1)
    assert polyval(p, xnew, x, y)[1].shape == (3, 2)

    # Multiple outputs: (n, k, 2)
    Y = np.column_stack([y, 3 * y])
    P, _, _ = regress(X, Y)
    yy, yint, _ = predict(X, Y, P, XX)
    assert yint.shape == (3, 2, 2)
    assert np.all(yint[..., 0] < yy) and np.all(yy < yint[..., 1])


@pytest.mark.parametrize("xnew", [np.array([5.0]), 5.0])
def test_nlpredict_single_point(xnew):
    """A length-1 or scalar xnew used to crash inside numdifftools."""
    x, y = _line_data()
    f = lambda x, m, b: m * x + b  # noqa: E731
    popt, _ = curve_fit(f, x, y)
    yp, yint, se = nlpredict(x, y, f, popt, xnew)
    ref = nlpredict(x, y, f, popt, np.array([5.0, 6.0]))
    assert yint.shape == (1, 2)
    np.testing.assert_allclose(yint[0], ref[1][0], rtol=1e-8)
    np.testing.assert_allclose(se, ref[2][:1], rtol=1e-8)
