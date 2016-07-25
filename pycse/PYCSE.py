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

# ** Boundary value solvers

# *** Linear BVP


def bvp_L0(p, q, r, x0, xL, alpha, beta, npoints=100):
    """Solve the linear BVP with constant boundary conditions.

    \\begin{equation}
    y'' + p(x)y' + q(x)y = r(x) \\
    y(x0) = \\alpha \\
    y(xL) = \\beta
    \end{equation}

    Parameters
    ----------
    p : a function p(x)
    q : a function q(x)
    r : a function r(x)
    x0 : The left-most boundary
    xL : The right-most boundary
    alpha : y(x0), it is constant
    beta : y(xL), it is constant
    npoints : int
        The number of points to discretize between x0 and xL.

    Example
    -------
    Solve y''(x) = -100

    >>> def p(x): return 0
    >>> def q(x): return 0
    >>> def r(x): return -100
    >>> x1 = 0; alpha = 0.0
    >>> x2 = 0.1; beta = 0.0
    >>> npoints = 100

    >>> x, y = bvp_L0(p, q, r, x1, x2, alpha, beta, npoints=100)
    >>> print(len(x))
    100
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> p = plt.plot(x, y)
    >>> plt.savefig('images/bvp-pycse.png')

    [[./images/bvp-pycse.png]]

    Returns
    -------
    [X, Y] where X is the vector of x-values where solutions exist,
    and Y is the solutions at those points.

    """

    h = (xL - x0) / (npoints - 3)
    X = np.linspace(x0 + h, xL - h, npoints - 2)

    # we do not do endpoints on A
    A = np.zeros((npoints - 2, npoints - 2))

    # special end point cases
    A[0][0] = -2.0 + h**2 * q(X[0])
    A[0][1] = 1.0 + h / 2.0 * p(X[0])
    A[-1][-2] = 1.0 - h / 2.0 * p(X[-1])
    A[-1][-1] = -2.0 + h**2 * q(X[-1])

    b = np.zeros(npoints - 2)
    b[0] = h**2 * r(X[0]) - alpha * (1.0 - h / 2.0 * p(X[0]))
    b[-1] = h**2 * r(X[-1]) - beta * (1.0 - h / 2.0 * p(X[-1]))

    # now fill in the matrix
    for i in range(1, npoints - 3):
        A[i][i-1] = 1.0 - h / 2.0 * p(X[i])
        A[i][i] = -2.0 + h**2 * q(X[i])
        A[i][i + 1] = 1 + h / 2.0 * p(X[i])

        b[i] = h**2 * r(X[i])

    # solve the equations
    y = np.linalg.solve(A, b)

    # add the boundary values back to the solution.
    return np.hstack([x0, X, xL]), np.hstack([alpha, y, beta])


def BVP_sh(F, x1, x2, alpha, beta, init):
    """A shooting method to solve odes.

    \\begin{equation}
    y'(x) = F(x, y)
    y(x1) = \\alpha
    y(x2) = \\beta
    \end{equation}

    Parameters
    ----------
    F : function of first order derivatives.
    x0 :
    xL :
    alpha : y(x1)
    beta : y(x2)
    init : A guess for y2(x1)

    y' is a system of ODES
    y1' = f(x, y1, y2)
    y2' = g(x, y1, y2)

    Returns
    -------
    The solution vectors [X, Y] where X is the vector of x-values for
    solutions, and Y is the array of solutions.

    """

    X = np.linspace(x1, x2)

    def objective(y20):
        y = odeint(F, [alpha, y20], X)
        y2 = y[:, 0]
        return beta - y2[-1]

    y20, = fsolve(objective, init)

    Y = odeint(F, [alpha, y20], X)
    return X, Y


# ** Nonlinear BVP_nl


def BVP_nl(F, X, BCS, init, **kwargs):
    """Solve nonlinear BVP \(y''(x) = F(x, y, y')\)
    with boundary conditions at y(x1) = a and y(x2) = b.

    Parameters
    ----------
    F : Function f(x, y, yprime)

    X : Uniformly spaced array from x1 to x2.

    BCS : Function that is zero at the boundary conditions. BCS(X, Y)

    init : Vector of initial guess solution values at the X points.

    kwargs are passed to :func:`scipy.optimize.fsolve`.

    Example
    -------
    Solve \(y'' = 3 y y'\) with boundary conditions y(0) = 0, and y(2) = 1.

    >>> import numpy as np
    >>> x1 = 0.0; x2 = 2.0
    >>> alpha = 0.0; beta = 1.0
    >>> def Ypp(x, y, yprime):
    ...    '''define y'' = 3*y*y' '''
    ...    return -3.0 * y * yprime
    ...
    >>> def BC(X, Y):
    ...    return [alpha - Y[0], beta - Y[-1]]
    ...
    >>> X = np.linspace(x1, x2)
    >>> init = alpha + (beta - alpha) / (x2 - x1) * X
    >>> y = BVP_nl(Ypp, X, BC, init)
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> p = plt.plot(X, y)
    >>> plt.savefig('images/bvp-nonlinear-2.png')

    [[./images/bvp-nonlinear-2.png]]

    Returns
    -------
    The solution array: Y

    """

    x1, x2 = X[0], X[-1]
    N = len(X)
    h = (x2 - x1) / (N - 1)

    def residuals(y):
        """When we have the right values of y, this function will be zero."""

        res = np.zeros(y.shape)

        a, b = BCS(X, y)

        res[0] = a

        for i in range(1, N - 1):
            x = X[i]
            YPP = (y[i - 1] - 2 * y[i] + y[i + 1]) / h**2
            YP = (y[i + 1] - y[i - 1]) / (2 * h)
            res[i] = YPP - F(x, y[i], YP)

        res[-1] = b
        return res

    Y, something, flag, msg = fsolve(residuals, init, full_output=1, **kwargs)
    if flag != 1:
        print(flag, msg)
        raise Exception(msg)
    return Y


def bvp_sh(odefun, bcfun, xspan, y0_guess):
    """Solve \(Y' = f(Y, x)\) by shooting method.

    Parameters
    ----------
    odefun : a function
    bcfun : bcfun(Ya, Yb) = 0
    xspan :
    y0_guess : Initial guess of the solution.

    Returns
    -------
    The solution array: Y
    """

    def objective(yinit):
        sol = odeint(odefun, yinit, xspan)
        res = bcfun(sol[0, :], sol[-1, :])
        return res

    Y0 = fsolve(objective, y0_guess)

    Y = odeint(odefun, Y0, xspan)
    return Y


def bvp(odefun, bcfun, X, yinit):
    """Solve \(Y' = f(Y, x)\).

    Parameters
    ----------
    odefun : \(f(Y, x)\)

    bcfun : function that returns the residuals at the boundary
        conditions, \(g(ya, yb)\)

    X : is a grid to discretize the region
    yinit :  guess of the solution on that grid

    Example:
    ========
    y'' + |y| = 0
    y(0) = 0
    y(4) = -2

    >>> def odefun(Y, x):
    ...     y1, y2 = Y
    ...     dy1dx = y2
    ...     dy2dx = -np.abs(y1)
    ...     return [dy1dx, dy2dx]
    ...
    >>> def bcfun(Y):
    ...     Ya = Y[0, :]
    ...     Yb = Y[-1, :]
    ...     y1a, y2a = Ya
    ...     y1b, y2b = Yb
    ...     return [y1a, -2 - y1b]
    ...
    >>> x = np.linspace(0, 4, 100)
    >>> y1 = x**0
    >>> y2 = 0.0 * x
    >>> Yinit = np.column_stack([y1, y2])
    >>> sol = bvp(odefun, bcfun, x, Yinit)

    Returns
    -------
    Solution array.

    """

    # we have to setup a series of equations equal to zero at the solution Y we
    # will end up with nX * neq equations. at each x point between the
    # boundaries we approximate the derivative by central difference neq of
    # these will be the boundary conditions neq * (nX - 1) of them will be the
    # derivatives at the interior points
    def objective(Yflat):
        Y = Yflat.reshape(yinit.shape)

        nX, neq = Y.shape

        # these are ultimately the "zeros" we solve for
        # The first set are the boundary conditions
        res = bcfun(Y)

        ode = np.array(odefun(Y, X))  # evaluate odefun for this Y

        # now we estimate dYdt from the solution and subtract it from ode
        # that will be zero at the solution
        # Y is nX rows by neq columns
        # I need column wise, centered finite difference from 1 to nX-2

        dYdt = (Y[2:, :] - Y[0:-2, :]) / (X[2:, np.newaxis] -
                                          X[0:-2, np.newaxis])

        Z = ode[1:-1, :] - dYdt

        res += Z.flat

        # we need to estimate the derivatives at one end point We use a 3 point
        # formula to estimate this derivative at the left boundary
        yjprime = (-3.0 * Y[0, :] + 4.0 * Y[1, :] - Y[2, :]) / (X[2] - X[0])
        z = ode[0, :] - yjprime
        res += z.flat

        return res

    solflat = fsolve(objective, yinit.flat)

    sol = solflat.reshape(yinit.shape)

    return sol


# * Numerical methods
def deriv(x, y, method='two-point'):
    """Compute the numerical derivative dy/dx for y(x)
    represented as arrays of y and x.

    Parameters
    ----------
    x : list/array of x-points
    y : list/array of y-values corresponding to y(x)
    method : string
        Possible values are:
        'two-point': centered difference
        'four-point': centered 4-point difference
        'fft' is an experimental method usind fft.

    Example
    -------
    >>> x = [0, 1, 2]
    >>> y = [0, 2, 4]
    >>> deriv(x, y)
    array([ 2.,  2.,  2.])

    Returns
    -------
    an array the same size as x and y.

    """

    x = np.array(x)
    y = np.array(y)
    if method == 'two-point':
        dydx = np.zeros(y.shape, np.float)  # we know it will be this size
        dydx[1:-1] = (y[2:] - y[0:-2]) / (x[2:] - x[0:-2])

        # now the end points
        dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return dydx

    elif method == 'four-point':
        dydx = np.zeros(y.shape, np.float)  # we know it will be this size
        h = x[1] - x[0]  # this assumes the points are evenly spaced!
        dydx[2:-2] = (y[0:-4] - 8 * y[1:-3] + 8 * y[3:-1] - y[4:]) / (12.0 * h)

        # simple differences at the end-points
        dydx[0] = (y[1] - y[0])/(x[1] - x[0])
        dydx[1] = (y[2] - y[1])/(x[2] - x[1])
        dydx[-2] = (y[-2] - y[-3]) / (x[-2] - x[-3])
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return dydx

    elif method == 'fft':
        # for this method, we have to start at zero.
        # we translate our variable to zero, perform the calculation, and then
        # translate back
        xp = np.array(x)
        xp -= x[0]

        N = len(xp)
        L = xp[-1]

        if N % 2 == 0:
            k = np.asarray(range(0, N / 2) + [0] + range(-N / 2 + 1, 0))
        else:
            k = np.asarray(range(0, (N - 1) / 2) + [0] +
                           range(-(N - 1) / 2, 0))

        k *= 2 * np.pi / L

        fd = np.fft.ifft(1.0j * k * np.fft.fft(y))
        return np.real(fd)


# * End
if __name__ == '__main__':
    import doctest
    doctest.testmod()
