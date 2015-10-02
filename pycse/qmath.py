'''
Module for functions that are wrapped to use quantities.
'''

import quantities as u
import numpy as np

from scipy.optimize import fsolve as _fsolve
from scipy.integrate import odeint as _odeint


def fsolve(func, t0, args=(),
           fprime=None, full_output=0, col_deriv=0,
           xtol=1.49012e-08, maxfev=0, band=None,
           epsfcn=0.0, factor=100, diag=None):
    '''wrapped fsolve command to work with units. We get the units on
    the function argument, then wrap the function so we can add units
    to the argument and return floats. Finally we call the original
    fsolve from scipy. '''

    try:
        # units on initial guess, normalized
        tU = [t / float(t) for t in t0]
    except TypeError:
        tU = t0 / float(t0)

    def wrapped_func(t, *args):
        't will be unitless, so we add unit to it. t * tU has units.'
        try:
            T = [x1 * x2 for x1, x2 in zip(t, tU)]
        except TypeError:
            T = t * tU

        try:
            return [float(x) for x in func(T, *args)]
        except TypeError:
            return float(func(T))

    sol = _fsolve(wrapped_func, t0, args,
                  fprime, full_output, col_deriv,
                  xtol, maxfev, band,
                  epsfcn, factor, diag)

    if full_output:
        x, infodict, ier, mesg = sol
        try:
            x = [x1 * x2 for x1, x2 in zip(x, tU)]
        except TypeError:
            x = x * tU
        return x, infodict, ier, mesg
    else:
        try:
            x = [x1 * x2 for x1, x2 in zip(sol, tU)]
        except TypeError:
            x = sol * tU
        return x


def odeint(func, y0, t, args=(),
           Dfun=None, col_deriv=0, full_output=0,
           ml=None, mu=None, rtol=None, atol=None,
           tcrit=None, h0=0.0, hmax=0.0, hmin=0.0,
           ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
           mxords=5, printmessg=0):
    "wrapper for scipy.integrate.odeint to work with quantities."

    def wrapped_func(Y0, T, *args):
        # put units on T if they are on the original t
        # check for units so we don't put them on twice
        if not hasattr(T, 'units') and hasattr(t, 'units'):
            T = T * t.units

        # now for the dependent variable units. Y0 may be a scalar or
        # a list or an array. we want to check each element of y0 for
        # units, and add them to the corresponding element of Y0 if we
        # need to.
        try:
            uY0 = [x for x in Y0]  # a list copy of contents of Y0
            # this works if y0 is an iterable, eg. a list or array
            for i, yi in enumerate(y0):
                if not hasattr(uY0[i], 'units') and hasattr(yi, 'units'):

                    uY0[i] = uY0[i] * yi.units

        except TypeError:
            # we have a scalar
            if not hasattr(Y0, 'units') and hasattr(y0, 'units'):
                uY0 = Y0 * y0.units

        # It is necessary to rescale this to prevent issues with non-simplified
        # units.
        val = func(uY0, t, *args).rescale(y0.units / t.units)

        try:
            return np.array([float(x) for x in val])
        except TypeError:
            return float(val)

    if full_output:
        y, infodict = _odeint(wrapped_func, y0, t, args,
                              Dfun, col_deriv, full_output,
                              ml, mu, rtol, atol,
                              tcrit, h0, hmax, hmin,
                              ixpr, mxstep, mxhnil, mxordn,
                              mxords, printmessg)
    else:
        y = _odeint(wrapped_func, y0, t, args,
                    Dfun, col_deriv, full_output,
                    ml, mu, rtol, atol,
                    tcrit, h0, hmax, hmin,
                    ixpr, mxstep, mxhnil, mxordn,
                    mxords, printmessg)

    # now we need to put units onto the solution units should be the
    # same as y0. We cannot put mixed units in an array, so, we return a list
    m, n = y.shape  # y is an ndarray, so it has a shape
    if n > 1:  # more than one equation, we need a list
        uY = [0 for yi in range(n)]

        for i, yi in enumerate(y0):
            if not hasattr(uY[i], 'units') and hasattr(yi, 'units'):
                uY[i] = y[:, i] * yi.units
            else:
                uY[i] = y[:, i]

    else:
        uY = y * y0.units

    y = uY

    if full_output:
        return y, infodict
    else:
        return y


if __name__ == '__main__':
    # Problem 1
    CA0 = 1 * u.mol / u.L
    CA = 0.01 * u.mol / u.L
    k = 1.0 / u.s

    def func(t):
        return CA - CA0 * np.exp(-k * t)

    tguess = 4 * u.s
    sol1, = fsolve(func, tguess)
    print 'sol1 = ', sol1

    # Problem 2
    def func2(X):
        a, b = X
        return [a**2 - 4 * u.kg**2,
                b**2 - 25 * u.J**2]

    Xguess = [2.2*u.kg, 5.2*u.J]
    sol, infodict, ier, mesg = fsolve(func2, Xguess, full_output=1)
    s2a, s2b = sol
    print 's2a = {0}\ns2b = {1}'.format(s2a, s2b)

    # Problem 3 - with an arg
    def func3(a, arg):
        return a**2 - 4*u.kg**2 + arg**2

    Xguess = 1.5 * u.kg
    arg = 0.0 * u.kg

    sol3, = fsolve(func3, Xguess, args=(arg,))
    print'sol3 = ', sol3

    ##################################################################
    # test a single ODE
    k = 0.23 / u.s
    Ca0 = 1 * u.mol / u.L

    def dCadt(Ca, t):
        return -k * Ca

    tspan = np.linspace(0, 5) * u.s
    sol = odeint(dCadt, Ca0, tspan)

    print sol[-1]

    import matplotlib.pyplot as plt
    plt.plot(tspan, sol)
    plt.xlabel('Time ({0})'.format(tspan.dimensionality.latex))
    plt.ylabel('$C_A$ ({0})'.format(sol.dimensionality.latex))
    plt.show()

    ##################################################################
    # test coupled ODEs
    lbmol = 453.59237 * u.mol

    kprime = 0.0266 * lbmol / u.hr / u.lb
    Fa0 = 1.08 * lbmol / u.hr
    alpha = 0.0166 / u.lb
    epsilon = -0.15

    def dFdW(F, W, alpha0):
        X, y = F
        dXdW = kprime / Fa0 * (1.0 - X)/(1.0 + epsilon * X) * y
        dydW = - alpha0 * (1.0 + epsilon * X) / (2.0 * y)
        return [dXdW, dydW]

    X0 = 0.0 * u.dimensionless
    y0 = 1.0

    # initial conditions
    F0 = [X0, y0]  # one without units, one with units, both are dimensionless

    wspan = np.linspace(0, 60) * u.lb

    sol = odeint(dFdW, F0, wspan, args=(alpha,))
    X, y = sol

    print 'Test 2'
    print X[-1]
    print y[-1]

    plt.figure()
    plt.plot(wspan, X, wspan, y)
    plt.legend(['X', '$P/P_0$'])
    plt.xlabel('Catalyst weight ({0})'.format(wspan.dimensionality.latex))
    plt.show()
