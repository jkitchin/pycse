"""Test module for PYCSE.py."""

import numpy as np
from pycse.PYCSE import (
    polyfit,
    polyval,
    regress,
    predict,
    nlinfit,
    nlpredict,
    Rsquared,
    bic,
    lbic,
    ivp,
)


def test_polyfit():
    """Test on fitting a line.

    I don't know if there are good ways to test that bint, se are correct.
    """
    x = np.array([0, 1])
    y = np.array([0, 1])

    b, bint, se = polyfit(x, y, 1)

    assert np.isclose(b[0], 1.0)
    assert np.isclose(b[1], 0.0)


def test_regress():
    x = np.array([0, 1])
    y = np.array([0, 1])

    X = np.column_stack([x, x**0])

    b, bint, se = regress(X, y, 0.05)

    assert np.isclose(b[0], 1.0)
    assert np.isclose(b[1], 0.0)
    assert bint is not None
    assert se is not None


def test_regress_multi_output():
    """Test regress with multiple output variables.

    This tests the case where y has multiple columns (multiple outputs),
    which triggers different code paths in the covariance calculation.
    """
    x = np.linspace(0, 1, 10)
    # Two output variables: y1 = 2x + 1, y2 = 3x + 2
    y = np.column_stack([2 * x + 1, 3 * x + 2])

    X = np.column_stack([x, np.ones_like(x)])
    b, bint, se = regress(X, y, alpha=0.05)

    # Check shapes for multi-output
    assert b.shape == (2, 2)  # 2 parameters x 2 outputs
    assert se.shape == (2, 2)

    # Check fitted parameters for each output
    np.testing.assert_allclose(b[:, 0], [2, 1], rtol=1e-10)  # y1 = 2x + 1
    np.testing.assert_allclose(b[:, 1], [3, 2], rtol=1e-10)  # y2 = 3x + 2

    # Check that standard errors are positive
    assert np.all(se > 0)

    # Check confidence interval shapes
    assert bint.shape == (2, 2, 2)  # 2 params x 2 bounds x 2 outputs


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


# Tests for polyval
def test_polyval_linear():
    """Test polyval with linear polynomial."""
    # Fit y = 2x + 1
    x = np.array([0, 1, 2, 3])
    y = 2 * x + 1

    # Fit the polynomial
    p, _, _ = polyfit(x, y, 1)

    # Predict at new points
    xnew = np.array([0.5, 1.5, 2.5])
    ypred, yint, pred_se = polyval(p, xnew, x, y)

    # Check predictions
    expected = 2 * xnew + 1
    np.testing.assert_allclose(ypred, expected, rtol=1e-10)

    # Check confidence intervals have correct shape
    assert yint.shape == (2, len(xnew))

    # Check prediction standard errors are positive
    assert np.all(pred_se > 0)

    # Check confidence intervals contain predicted values
    assert np.all(yint[0] <= ypred)
    assert np.all(ypred <= yint[1])


def test_polyval_quadratic():
    """Test polyval with quadratic polynomial."""
    # Fit y = x^2 + 2x + 1
    x = np.array([0, 1, 2, 3, 4])
    y = x**2 + 2 * x + 1

    # Fit the polynomial
    p, _, _ = polyfit(x, y, 2)

    # Predict at new points
    xnew = np.array([0.5, 1.5, 2.5])
    ypred, yint, pred_se = polyval(p, xnew, x, y)

    # Check predictions
    expected = xnew**2 + 2 * xnew + 1
    np.testing.assert_allclose(ypred, expected, rtol=1e-10)

    # Check standard errors are positive
    assert np.all(pred_se > 0)


# Tests for predict
def test_predict_linear():
    """Test predict with linear regression."""
    # Simple linear case: y = 2x + 1
    x = np.array([0, 1, 2, 3])
    y = 2 * x + 1

    # Create design matrix
    X = np.column_stack([x, np.ones_like(x)])

    # Fit using regress
    pars, _, _ = regress(X, y)

    # Predict at new points
    xnew = np.array([0.5, 1.5, 2.5])
    XX = np.column_stack([xnew, np.ones_like(xnew)])

    ypred, yint, pred_se = predict(X, y, pars, XX)

    # Check predictions
    expected = 2 * xnew + 1
    np.testing.assert_allclose(ypred, expected, rtol=1e-10)

    # Check confidence intervals
    assert yint.shape == (2, len(xnew))
    assert np.all(yint[0] <= ypred)
    assert np.all(ypred <= yint[1])

    # Check standard errors are positive
    assert np.all(pred_se > 0)


def test_predict_with_noise():
    """Test predict with noisy data."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 50)

    X = np.column_stack([x, np.ones_like(x)])
    pars, _, _ = regress(X, y)

    # Predict at new points
    xnew = np.array([5.0, 7.5])
    XX = np.column_stack([xnew, np.ones_like(xnew)])

    ypred, yint, pred_se = predict(X, y, pars, XX)

    # Predictions should be close to true values
    expected = 2 * xnew + 1
    np.testing.assert_allclose(ypred, expected, rtol=0.1)

    # Confidence intervals should be wider with noisy data
    ci_width = yint[1] - yint[0]
    assert np.all(ci_width > 0)


def test_predict_multi_output():
    """Test predict with multiple output variables.

    This tests the case where we fit and predict multiple outputs,
    which triggers different code paths in the prediction calculation.
    """
    x = np.linspace(0, 1, 10)
    # Two output variables: y1 = 2x + 1, y2 = 3x + 2
    y = np.column_stack([2 * x + 1, 3 * x + 2])

    X = np.column_stack([x, np.ones_like(x)])
    pars, _, _ = regress(X, y, alpha=0.05)

    # Predict at new points
    xnew = np.array([0.25, 0.5, 0.75])
    XX = np.column_stack([xnew, np.ones_like(xnew)])

    ypred, yint, pred_se = predict(X, y, pars, XX)

    # Check shapes for multi-output predictions
    assert ypred.shape == (3, 2)  # 3 prediction points x 2 outputs

    # Check predictions for each output
    expected_y1 = 2 * xnew + 1
    expected_y2 = 3 * xnew + 2
    np.testing.assert_allclose(ypred[:, 0], expected_y1, rtol=1e-10)
    np.testing.assert_allclose(ypred[:, 1], expected_y2, rtol=1e-10)

    # Check that prediction standard errors are positive
    assert np.all(pred_se > 0)


# Tests for nlpredict
def test_nlpredict_linear():
    """Test nlpredict with linear model (should match predict)."""
    # Use linear model: y = 2x + 1
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x + 1

    def linear_model(x, m, b):
        return m * x + b

    # Fit using nlinfit
    popt, _, _ = nlinfit(linear_model, x, y, [1, 0])

    # Predict at new points
    xnew = np.array([0.5, 1.5, 2.5])
    ypred, yint, pred_se = nlpredict(x, y, linear_model, popt, xnew)

    # Check predictions
    expected = 2 * xnew + 1
    np.testing.assert_allclose(ypred, expected, rtol=1e-6)

    # Check confidence intervals
    assert yint.shape == (len(xnew), 2)
    assert np.all(yint[:, 0] <= ypred)
    assert np.all(ypred <= yint[:, 1])

    # Check standard errors are positive
    assert np.all(pred_se > 0)


def test_nlpredict_exponential():
    """Test nlpredict with exponential model."""
    # Use exponential model: y = a * exp(b * x)
    x = np.linspace(0, 2, 20)
    y = 2 * np.exp(0.5 * x)

    def exp_model(x, a, b):
        return a * np.exp(b * x)

    # Fit using nlinfit
    popt, _, _ = nlinfit(exp_model, x, y, [1, 1])

    # Predict at new points
    xnew = np.array([0.5, 1.0, 1.5])
    ypred, yint, pred_se = nlpredict(x, y, exp_model, popt, xnew)

    # Check predictions are close
    expected = 2 * np.exp(0.5 * xnew)
    np.testing.assert_allclose(ypred, expected, rtol=1e-5)

    # Check standard errors are positive
    assert np.all(pred_se > 0)


# Tests for bic
def test_bic_linear():
    """Test BIC for linear model."""
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x + 1

    def linear_model(x, m, b):
        return m * x + b

    popt, _, _ = nlinfit(linear_model, x, y, [1, 0])

    bic_value = bic(x, y, linear_model, popt)

    # BIC should be a real number
    assert np.isfinite(bic_value)

    # For perfect fit, BIC should be very negative (due to log(0) handling)
    # Actually, with perfect fit RSS=0, log(RSS/n) -> -inf
    # But in practice with numerical precision, it's a very negative number
    assert bic_value < 0


def test_bic_with_noise():
    """Test BIC with noisy data."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 50)

    def linear_model(x, m, b):
        return m * x + b

    popt, _, _ = nlinfit(linear_model, x, y, [1, 0])

    bic_value = bic(x, y, linear_model, popt)

    # BIC should be finite
    assert np.isfinite(bic_value)


def test_bic_model_comparison():
    """Test that BIC can compare models."""
    np.random.seed(42)
    x = np.linspace(0, 2, 30)
    # True model is quadratic
    y_true = x**2 + x + 1
    y = y_true + np.random.normal(0, 0.1, len(x))

    # Linear model (underfitting)
    def linear_model(x, a, b):
        return a * x + b

    popt_linear, _, _ = nlinfit(linear_model, x, y, [1, 1])
    bic_linear = bic(x, y, linear_model, popt_linear)

    # Quadratic model (correct)
    def quad_model(x, a, b, c):
        return a * x**2 + b * x + c

    popt_quad, _, _ = nlinfit(quad_model, x, y, [1, 1, 1])
    bic_quad = bic(x, y, quad_model, popt_quad)

    # Quadratic should have lower (better) BIC despite more parameters
    assert bic_quad < bic_linear


# Tests for lbic
def test_lbic_linear():
    """Test lbic for linear regression."""
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x + 1

    X = np.column_stack([x, np.ones_like(x)])
    popt, _, _ = regress(X, y)

    bic_value = lbic(X, y, popt)

    # BIC should be finite
    assert np.isfinite(bic_value)

    # For perfect fit, BIC should be very negative
    assert bic_value < 0


def test_lbic_with_noise():
    """Test lbic with noisy data."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 50)

    X = np.column_stack([x, np.ones_like(x)])
    popt, _, _ = regress(X, y)

    bic_value = lbic(X, y, popt)

    # BIC should be finite
    assert np.isfinite(bic_value)


def test_lbic_model_comparison():
    """Test that lbic can compare models."""
    np.random.seed(42)
    x = np.linspace(0, 2, 30)
    # True model is quadratic
    y = x**2 + 2 * x + 1 + np.random.normal(0, 0.1, len(x))

    # Linear model (underfitting)
    X_linear = np.column_stack([x, np.ones_like(x)])
    popt_linear, _, _ = regress(X_linear, y)
    bic_linear = lbic(X_linear, y, popt_linear)

    # Quadratic model (correct)
    X_quad = np.column_stack([x**2, x, np.ones_like(x)])
    popt_quad, _, _ = regress(X_quad, y)
    bic_quad = lbic(X_quad, y, popt_quad)

    # Quadratic should have lower (better) BIC
    assert bic_quad < bic_linear
