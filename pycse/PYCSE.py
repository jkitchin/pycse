"""Module containing useful scientific and engineering functions.

- Linear regression
- Nonlinear regression.
- Differential equation solvers.

See http://kitchingroup.cheme.cmu.edu/pycse

Copyright 2020, John Kitchin
(see accompanying license files for details).
"""

import warnings
import numpy as np
from scipy.stats.distributions import t
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

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
    
def regress(A, y, alpha=None, *args, **kwargs):
    """Linear least squares regression with confidence intervals.

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

    ags and kwargs are passed to np.linalg.lstsq

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
      bint is a 2D array of confidence intervals
      se is an array of standard error for each parameter.

    """

    # This is to silence an annoying FutureWarning.
    if 'rcond' not in kwargs:
        kwargs['rcond'] = None
        
    b, res, rank, s = np.linalg.lstsq(A, y, *args, **kwargs)

    bint, se = None, None

    if alpha is not None:
        # compute the confidence intervals
        n = len(y)
        k = len(b)

        errors = y - np.dot(A, b)
        sigma2 = np.sum(errors**2) / (n - k)  # RMSE

        covariance = np.linalg.inv(np.dot(A.T, A))

        C = sigma2 * covariance
        dC = np.diag(C)

        if (dC < 0.0).any():
            warnings.warn('\n{0}\ndetected a negative number in your'
                          'covariance matrix. Taking the absolute value'
                          'of the diagonal. something is probably wrong'
                          'with your data or model'.format(dC))
            dC = np.abs(dC)

        se = np.sqrt(dC)  # standard error

        sT = t.ppf(1.0 - alpha/2.0, n - k - 1)  # student T multiplier
        CI = sT * se

        bint = np.array([(beta - ci, beta + ci) for beta, ci in zip(b, CI)])

    return (b, bint, se)

# * Nonlinear regression


def nlinfit(model, x, y, p0, alpha=0.05, **kwargs):
    """Nonlinear regression with confidence intervals.

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
    (array([  2.00000000e+00,  -2.18062024e-12]), array([[  2.00000000e+00,   2.00000000e+00],
           [ -2.18315458e-12,  -2.17808591e-12]]), array([  1.21903752e-12,   1.99456367e-16]))

    Returns
    -------
    [p, pint, SE]
      p is an array of the fitted parameters
      pint is an array of confidence intervals
      SE is an array of standard errors for the parameters.

    """
    pars, pcov = curve_fit(model, x, y, p0=p0, **kwargs)
    n = len(y)    # number of data points
    p = len(pars)  # number of parameters

    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    tval = t.ppf(1.0-alpha/2., dof)

    SE = []
    pint = []
    for i, p, var in zip(range(n), pars, np.diag(pcov)):
        sigma = var**0.5
        SE.append(sigma)
        pint.append([p - sigma * tval, p + sigma * tval])

    return (pars, np.array(pint), np.array(SE))

# * ivp


def ivp(f, tspan, y0, *args, **kwargs):
    """A friendlier version of solve_ivp.

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
    if 'max_step' not in kwargs:
        kwargs['max_step'] = min(np.diff(tspan))

    if 'dense_output' not in kwargs:
        kwargs['dense_output'] = True

    if 't_eval' not in kwargs:
        kwargs['t_eval'] = tspan

    sol = solve_ivp(f, (t0, tf), y0, *args, **kwargs)

    if sol.status != 0:
        print(sol.message)

    return sol


# * End


if __name__ == '__main__':
    import doctest
    doctest.testmod()
