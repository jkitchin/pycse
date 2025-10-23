"""Module containing useful scientific and engineering functions.

- Linear regression
- Nonlinear regression.
- Differential equation solvers.

See http://kitchingroup.cheme.cmu.edu/pycse

Copyright 2025, John Kitchin
(see accompanying license files for details).
"""

# pylint: disable=invalid-name

import warnings
import numpy as np
from scipy.stats.distributions import t
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

import numdifftools as nd

# * Linear regression


def polyfit(x, y, deg, alpha=0.05, *args, **kwargs):
    """Least squares polynomial fit with parameter confidence intervals.

    Parameters
    ----------
    x : array_like, shape (M,)
      x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
      y-coordinates of the sample points. Several data sets of sample
      points sharing the same x-coordinates can be fitted at once by
      passing in a 2D-array that contains one dataset per column.
    deg : int
      Degree of the fitting polynomial
    *args and **kwargs are passed to regress.

    Returns
    -------
      [b, bint, se]
      b is a vector of the fitted parameters
      bint is a 2D array of confidence intervals
      se is an array of standard error for each parameter.
    """
    # in vander, the second arg is the number of columns, so we have to add one
    # to the degree since there are N + 1 columns for a polynomial of order N
    X = np.vander(x, deg + 1)
    return regress(X, y, alpha, *args, **kwargs)


def polyval(p, x, X, y, alpha=0.05, ub=1e-5, ef=1.05):
    """Evaluate the polynomial p at x with prediction intervals.

    Parameters
    ----------
    p: parameters from pycse.polyfit
    x: array_like, shape (M,)
      x-coordinates to evaluate the polynomial at.
    X: array_like, shape (N,)
      the original x-data that p was fitted from.
    y: array-like, shape (N,)
      the original y-data that p was fitted from.

        alpha : confidence level, 95% = 0.05
    ub : upper bound for smallest allowed Hessian eigenvalue
    ef : eigenvalue factor for scaling Hessian

    Returns
    -------
    y, yint, pred_se
    y : the predicted values
    yint: confidence interval
    """
    deg = len(p) - 1
    _x = np.vander(x, deg + 1)  # to predict at

    _X = np.vander(X, deg + 1)  # original data

    return predict(_X, y, p, _x, alpha, ub, ef)


def regress(A, y, alpha=0.05, *args, **kwargs):
    r"""Linear least squares regression with confidence intervals.

    Solve the matrix equation \(A p = y\) for p.

    The confidence intervals account for sample size using a student T
    multiplier.

    This code is derived from the descriptions at
    http://www.weibull.com/DOEWeb/confidence_intervals_in_multiple_linear_regression.htm
    and
    http://www.weibull.com/DOEWeb/estimating_regression_models_using_least_squares.htm

    Parameters
    ----------
    A : a matrix of function values in columns, e.g.
        A = np.column_stack([T**0, T**1, T**2, T**3, T**4])

    y : a vector of values you want to fit

    alpha : 100*(1 - alpha) confidence level

    args and kwargs are passed to np.linalg.lstsq

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 2, 4])
    >>> X = np.column_stack([x**0, x])
    >>> regress(X, y)
    (array([ -5.12790050e-16,   2.00000000e+00]), None, None)

    Returns
    -------
      [b, bint, se]
      b is a vector of the fitted parameters
      bint is an array of confidence intervals. The ith row is for the ith parameter.
      se is an array of standard error for each parameter.

    """
    # This is to silence an annoying FutureWarning.
    if "rcond" not in kwargs:
        kwargs["rcond"] = None

    b, _, _, _ = np.linalg.lstsq(A, y, *args, **kwargs)

    bint, se = None, None

    if alpha is not None:
        # compute the confidence intervals
        n = len(y)
        k = len(b)

        errors = y - np.dot(A, b)  # this may have many columns
        sigma2 = np.sum(errors**2, axis=0) / (n - k)  # RMSE

        covariance = np.linalg.inv(np.dot(A.T, A))

        # sigma2 is either a number, or (1, noutputs)
        # covariance is npars x npars
        # I need to scale C for each column
        try:
            C = [covariance * s for s in sigma2]
            dC = np.array([np.diag(c) for c in C]).T
        except TypeError:
            C = covariance * sigma2
            dC = np.diag(C)

        # The diagonal on each column is related to the standard error

        if (dC < 0.0).any():
            warnings.warn(
                "\n{0}\ndetected a negative number in your"
                " covariance matrix. Taking the absolute value"
                " of the diagonal. something is probably wrong"
                " with your data or model".format(dC)
            )
            dC = np.abs(dC)

        se = np.sqrt(dC)  # standard error

        # CORRECTED: Use n - k degrees of freedom (not n - k - 1)
        # For linear regression with n observations and k parameters,
        # the residual has exactly n - k degrees of freedom.
        sT = t.ppf(1.0 - alpha / 2.0, n - k)  # student T multiplier
        CI = sT * se

        # bint is a little tricky, and depends on the shape of the output.
        bint = np.array([(b - CI, b + CI)]).T

    return (b, bint.squeeze(), se)


def predict(X, y, pars, XX, alpha=0.05, ub=1e-5, ef=1.05):
    """Prediction interval for linear regression.

    Based on the delta method.

    Parameters
    ----------
    X : known x-value array, one row for each y-point
    y : known y-value array
    pars : fitted parameters
    XX : x-value array to make predictions for
    alpha : confidence level, 95% = 0.05
    ub : upper bound for smallest allowed Hessian eigenvalue
    ef : eigenvalue factor for scaling Hessian

    See https://en.wikipedia.org/wiki/Prediction_interval#Unknown_mean,_unknown_variance

    Returns
    y, yint, pred_se
    y : the predicted values
    yint: confidence interval
    pred_se: std error on predictions.
    """
    n = len(X)
    npars = len(pars)
    dof = n - npars

    errs = y - X @ pars

    sse = np.sum(errs**2, axis=0)

    # CORRECTED: Use unbiased variance estimator with correct DOF
    # mse represents σ², the noise variance
    mse = sse / dof  # Was: sse / n

    gprime = XX

    # CORRECTED: Removed factor of 2 for covariance calculation
    # Even though np.linalg.lstsq minimizes SSE (with Hessian H = 2X'X),
    # the Fisher Information is I = H / (2σ²) = 2X'X / (2σ²) = X'X / σ²
    # Therefore: Cov(β) = I⁻¹ = σ² × (X'X)⁻¹ (factor of 2 cancels)
    # This matches what regress() correctly uses (line 142)
    hat = X.T @ X  # Was: 2 * X.T @ X
    eps = max(ub, ef * np.linalg.eigvals(hat).min())

    # Parameter covariance matrix
    I_fisher = np.linalg.pinv(hat + np.eye(npars) * eps)

    # CORRECTED: Compute parameter uncertainty and total prediction uncertainty separately
    # Parameter uncertainty: SE(X̂β) = sqrt(σ² × x'(X'X)⁻¹x)
    # Total prediction uncertainty: SE(ŷ - y_new) = sqrt(σ² + SE(X̂β)²)
    # The old formula used (1 + 1/n)^0.5 approximation, which only holds at the sample mean

    try:
        # This happens if mse is iterable
        param_se = np.sqrt([_mse * np.diag(gprime @ I_fisher @ gprime.T) for _mse in mse]).T
        # Total prediction SE: sqrt(noise_variance + parameter_variance)
        total_se = np.sqrt([_mse + _param_se**2 for _mse, _param_se in zip(mse, param_se.T)]).T
    except TypeError:
        # This happens if mse is a single number
        # you need at least 1d to get a diagonal. This line is needed because
        # there is a case where there is one prediction where this product leads
        # to a scalar quantity and we need to upgrade it to 1d to avoid an
        # error.
        gig = np.atleast_1d(gprime @ I_fisher @ gprime.T)
        param_se = np.sqrt(mse * np.diag(gig)).T
        # Total prediction SE includes both noise and parameter uncertainty
        total_se = np.sqrt(mse + param_se**2)

    tval = t.ppf(1.0 - alpha / 2.0, dof)

    yy = XX @ pars

    # Prediction intervals using total uncertainty
    yint = np.array(
        [
            yy - tval * total_se,
            yy + tval * total_se,
        ]
    )

    return (yy, yint, total_se)


# * Nonlinear regression


def nlinfit(model, x, y, p0, alpha=0.05, **kwargs):
    r"""Nonlinear regression with confidence intervals.

    Parameters
    ----------
    model : function f(x, p0, p1, ...) = y
    x : array of the independent data
    y : array of the dependent data
    p0 : array of the initial guess of the parameters
    alpha : 100*(1 - alpha) is the confidence interval
        i.e. alpha = 0.05 is 95% confidence

    kwargs are passed to curve_fit.

    Example
    -------
    Fit a line \(y = mx + b\) to some data.

    >>> import numpy as np
    >>> def f(x, m, b):
    ...    return m * x + b
    ...
    >>> X = np.array([0, 1, 2])
    >>> y = np.array([0, 2, 4])
    >>> nlinfit(f, X, y, [0, 1])
    (array([  2.00000000e+00,  -2.18062024e-12]),
     array([[  2.00000000e+00,   2.00000000e+00],
           [ -2.18315458e-12,  -2.17808591e-12]]),
     array([  1.21903752e-12,   1.99456367e-16]))

    Returns
    -------
    [p, pint, SE]
      p is an array of the fitted parameters
      pint is an array of confidence intervals
      SE is an array of standard errors for the parameters.

    """
    pars, pcov = curve_fit(model, x, y, p0=p0, **kwargs)
    n = len(y)  # number of data points
    p = len(pars)  # number of parameters

    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    tval = t.ppf(1.0 - alpha / 2.0, dof)

    SE = []
    pint = []
    for p, var in zip(pars, np.diag(pcov)):
        sigma = var**0.5
        SE.append(sigma)
        pint.append([p - sigma * tval, p + sigma * tval])

    return (pars, np.array(pint), np.array(SE))


def nlpredict(X, y, model, popt, xnew, loss=None, alpha=0.05, ub=1e-5, ef=1.05):
    """Prediction error for a nonlinear fit.

    Parameters
    ----------
    X : array-like
        Independent variable data used for fitting
    y : array-like
        Dependent variable data used for fitting
    model : callable
        Model function with signature model(x, ...)
    popt : array-like
        Optimized parameters from fitting (e.g., from curve_fit or nlinfit)
    xnew : array-like
        x-values to predict at
    loss : callable, optional
        Loss function that was minimized during fitting, with signature loss(*params).
        If None (default), assumes scipy.optimize.curve_fit was used and automatically
        constructs the correct loss function: loss = 0.5 * sum((y - model(X, *p))**2).
        This is the ½SSE convention used by scipy's least_squares optimizer.
        If you used a different optimizer or loss function, provide it explicitly.
    alpha : float, optional
        Confidence level (default: 0.05 for 95% confidence intervals)
    ub : float, optional
        Upper bound for smallest allowed Hessian eigenvalue (default: 1e-5)
    ef : float, optional
        Eigenvalue factor for scaling Hessian (default: 1.05)

    This function uses numdifftools for the Hessian and Jacobian.

    Returns
    -------
    y : array
        Predicted values at xnew
    yint : array
        Prediction intervals at alpha confidence level, shape (n, 2)
    se : array
        Standard error of predictions

    Notes
    -----
    The default loss function (½SSE) matches the convention used by scipy.optimize.curve_fit,
    which internally minimizes 0.5 * sum(residuals**2). If you provide a custom loss function,
    ensure it uses the same convention as your fitting procedure.
    """
    # If no loss function provided, assume curve_fit was used (½SSE convention)
    if loss is None:

        def loss(*p):
            return 0.5 * np.sum((y - model(X, *p)) ** 2)

    ypred = model(xnew, *popt)

    hessp = nd.Hessian(lambda p: loss(*p))(popt)
    # for making the Hessian better conditioned.
    eps = max(ub, ef * np.linalg.eigvals(hessp).min())

    sse = loss(*popt)
    n = len(y)
    p = len(popt)
    mse = sse / (n - p)  # Use unbiased estimator
    I_fisher = np.linalg.pinv(hessp + np.eye(len(popt)) * eps)

    gprime = nd.Jacobian(lambda p: model(xnew, *p))(popt)

    sigmas = np.sqrt(mse * np.diag(gprime @ I_fisher @ gprime.T))
    tval = t.ppf(1 - alpha / 2, len(y) - len(popt))

    return [
        ypred,
        np.array(
            [
                # https://online.stat.psu.edu/stat501/lesson/7/7.2
                ypred - tval * (sigmas**2 + mse) ** 0.5,  # lower bound
                ypred + tval * (sigmas**2 + mse) ** 0.5,  # upper bound
            ]
        ).T,
        sigmas,
    ]


def Rsquared(y, Y):
    """Return R^2, or coefficient of determination.

    y is a 1d array of observations.
    Y is a 1d array of predictions from a model.

    Returns
    -------
    The R^2 value for the fit.
    """
    errs = y - Y
    SS_res = np.sum(errs**2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - SS_res / SS_tot


def bic(x, y, model, popt):
    """Compute the Bayesian information criterion (BIC).

    Parameters
    ----------
    model : function(x, ...) returns prediction for y
    popt : optimal parameters
    y : array, known y-values

    Returns
    -------
    BIC : float

    https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
    """
    n = len(y)
    k = len(popt)
    rss = np.sum((model(x, *popt) - y) ** 2)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


def lbic(X, y, popt):
    """Compute the Bayesian information criterion for a linear model.

    Paramters
    ---------
    X : array of input variables in column form
    y : known y values
    popt : fitted parameters

    Returns
    -------
    BIC : float
    """
    n = len(y)
    k = len(popt)
    rss = np.sum((X @ popt - y) ** 2)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


# * ivp
def ivp(f, tspan, y0, *args, **kwargs):
    """Solve an ODE initial value problem.

    This provides some convenience defaults that I think are better than
    solve_ivp.

    Parameters
    ----------
    f : function
    callable y'(x, y) = f(x, y)

    tspan : array
    The x points you want the solution at. The first and last points are used in
    tspan in solve_ivp.

    y0 : array
    Initial conditions

    *args : type
    arbitrary positional arguments to pass to solve_ivp

    **kwargs : type arbitrary kwargs to pass to solve_ivp.
    max_step is set to be the min diff of tspan. dense_output is set to True.
    t_eval is set to the array specified in tspan.

    Returns
    -------
    solution from solve_ivp

    """
    t0, tf = tspan[0], tspan[-1]

    # make the max_step the smallest step in tspan, or what is in kwargs.
    if "max_step" not in kwargs:
        kwargs["max_step"] = min(np.diff(tspan))

    if "dense_output" not in kwargs:
        kwargs["dense_output"] = True

    if "t_eval" not in kwargs:
        kwargs["t_eval"] = tspan

    sol = solve_ivp(f, (t0, tf), y0, *args, **kwargs)

    if sol.status != 0:
        print(sol.message)

    return sol


# * End


if __name__ == "__main__":
    import doctest

    doctest.testmod()
