"""Module containing useful scientific and engineering functions.

- Linear regression
- Nonlinear regression.
- Differential equation solvers.

See http://kitchingroup.cheme.cmu.edu/pycse

Copyright 2015, John Kitchin
(see accompanying license files for details).
"""

import warnings
import numpy as np
from scipy.stats.distributions import t
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import odeint

# * Linear regression


def regress(A, y, alpha=None):
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

    b, res, rank, s = np.linalg.lstsq(A, y)

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


def nlinfit(model, x, y, p0, alpha=0.05):
    """Nonlinear regression with confidence intervals.

    Parameters
    ----------
    model : function f(x, p0, p1, ...) = y
    x : array of the independent data
    y : array of the dependent data
    p0 : array of the initial guess of the parameters
    alpha : 100*(1 - alpha) is the confidence interval
        i.e. alpha = 0.05 is 95% confidence

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
    pars, pcov = curve_fit(model, x, y, p0=p0)
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

# * Differential equations
# ** Ordinary differential equations


def odelay(func, y0, xspan, events, TOLERANCE=1e-6,
           fsolve_args=None, **kwargs):
    """Solve an ODE with events.

    Parameters
    ----------
    func : y' = func(Y, x)
        func takes an independent variable x, and the Y value(s),
        and returns y'.

    y0 : The initial conditions at xspan[0].

    xspan : array to integrate the solution at.
        The initial condition is at xspan[0].

    events : list of callable functions with signature event(Y, x).
        These functions return zero when an event has happened.

        [value, isterminal, direction] = event(Y, x)

        value is the value of the event function. When value = 0, an event
        is triggered

        isterminal = True if the integration is to terminate at a zero of
        this event function, otherwise, False.

        direction = 0 if all zeros are to be located (the default), +1
        if only zeros where the event function is increasing, and -1 if
        only zeros where the event function is decreasing.

    TOLERANCE : float
        Used to identify when an event has occurred.

    fsolve_args : a dictionary of options for fsolve

    kwargs : Additional keyword options you want to send to odeint.

    Returns
    -------
    [x, y, te, ye, ie]
        x is the independent variable array
        y is the solution
        te is an array of independent variable values where events occurred
        ye is an array of the solution at the points where events occurred
        ie is an array of indices indicating which event function occurred.
    """
    if 'full_output' in kwargs:
        raise Exception('full_output not supported as an option')

    if fsolve_args is None:
        fsolve_args = {}

    x0 = xspan[0]  # initial point

    X = [x0]
    sol = [y0]
    TE, YE, IE = [], [], []  # to store where events occur

    # initial value of events
    e = np.zeros((len(events), len(xspan)))
    for i, event in enumerate(events):
        e[i, 0], isterminal, direction = event(y0, x0)

    # now we step through the integration
    for i, x1 in enumerate(xspan[0:-1]):
        x2 = xspan[i + 1]
        f1 = sol[i]

        f2 = odeint(func, f1, [x1, x2], **kwargs)

        X += [x2]
        sol += [f2[-1, :]]

        # check event functions. At each step we compute the event
        # functions, and check if they have changed sign since the
        # last step. If they changed sign, it implies a zero was
        # crossed.
        for j, event in enumerate(events):
            e[j, i + 1], isterminal, direction = event(sol[i + 1], X[i + 1])

            if ((e[j, i + 1] * e[j, i] < 0) or      # sign change in
                                                    # event means zero
                                                    # crossing
                np.abs(e[j, i + 1]) < TOLERANCE or  # this point is
                                                    # practically 0
                np.abs(e[j, i]) < TOLERANCE):

                xLt = X[-1]       # Last point
                fLt = sol[-1]

                # we need to find a value of x that makes the event zero
                def objective(x):
                    # evaluate ode from xLT to x
                    txspan = [xLt, x]
                    tempsol = odeint(func, fLt, txspan, **kwargs)
                    sol = tempsol[-1, :]
                    val, isterminal, direction = event(sol, x)
                    return val

                from scipy.optimize import fsolve

                # this should be the value of x that makes the event zero
                xZ, = fsolve(objective, xLt, **fsolve_args)

                # now evaluate solution at this point, so we can
                # record the function values here.
                txspan = [xLt, xZ]
                tempsol = odeint(func, fLt, txspan, **kwargs)
                fZ = tempsol[-1, :]

                vZ, isterminal, direction = event(fZ, xZ)

                COLLECTEVENT = False
                if direction == 0:
                    COLLECTEVENT = True
                elif (e[j, i + 1] > e[j, i]) and direction == 1:
                    COLLECTEVENT = True
                elif (e[j, i + 1] < e[j, i]) and direction == -1:
                    COLLECTEVENT = True

                if COLLECTEVENT:
                    TE.append(xZ)
                    YE.append(fZ)
                    IE.append(j)

                    if isterminal:
                        X[-1] = xZ
                        sol[-1] = fZ
                        return (np.array(X),
                                np.array(sol),
                                np.array(TE),
                                np.array(YE),
                                np.array(IE))

    # at the end, return what we have
    return (np.array(X),
            np.array(sol),
            np.array(TE),
            np.array(YE),
            np.array(IE))


# * End


if __name__ == '__main__':
    import doctest
    doctest.testmod()
