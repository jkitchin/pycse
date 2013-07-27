'''
Wrapped functions for handling units. This is using my pycse.units package.

Functions wrapped so far
========================

scipy
-----

fsolve
odeint
quad

numpy
-----

polyfit
polyder

Need to do
----------

regress
nlinfit

trapz
interp1
'''

from pycse.units import *
import numpy as np
from scipy.optimize import fsolve as _fsolve
from scipy.integrate import odeint as _odeint

def fsolve(func, x0, args=(),
           fprime=None, full_output=0, col_deriv=0,
           xtol=1.49012e-08, maxfev=0, band=None,
           epsfcn=0.0, factor=100, diag=None):
    '''
    Wrapped scipy.optimize.fsolve to handle units

    Presumably func and x0 have units associated with them.
    We simply wrap the function, 
    '''

    def wrappedfunc(x, *args):
        x = Unit(x, x0.exponents, x0.label)
        return float(func(x))
    
    if full_output:
        x, infodict, ier, mesg = _fsolve(wrappedfunc, float(x0), args,
                                         fprime, full_output, col_deriv,
                                         xtol, maxfev, band,
                                         epsfcn, factor, diag)
        x = Unit(x, x0.exponents, x0.label)
        return x, infodict, ier, mesg
    else:
        x, = _fsolve(wrappedfunc, float(x0), args,
                    fprime, full_output, col_deriv,
                    xtol, maxfev, band,
                    epsfcn, factor, diag)
        
        x = Unit(x, x0.exponents, x0.label)
        return x
    

def odeint(func, y0, tspan, args=(), Dfun=None, col_deriv=0,
    full_output=0, ml=None, mu=None, rtol=None, atol=None, tcrit=None,
    h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
    mxords=5, printmessg=0):
    'Wrapped scipy.integrate.odeint with units'

    def wrappedfunc(y, t, *args):
        y = Unit(y, y0.exponents, y0.label)
        t = Unit(tspan, tspan.exponents, tspan.label)
        return float(func(y, t, *args))

    if full_output:
        y, infodict = _odeint(wrappedfunc, y0, tspan, args=(),
                              Dfun=None, col_deriv=0, full_output=0,
                              ml=None, mu=None, rtol=None, atol=None,
                              tcrit=None, h0=0.0, hmax=0.0, hmin=0.0,
                              ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
                              mxords=5, printmessg=0)
    else:
        y = _odeint(wrappedfunc, y0, tspan, args=(),
                              Dfun=None, col_deriv=0, full_output=0,
                              ml=None, mu=None, rtol=None, atol=None,
                              tcrit=None, h0=0.0, hmax=0.0, hmin=0.0,
                              ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
                              mxords=5, printmessg=0)
    y = Unit(y, y0.exponents, y0.label)
    return y


def polyfit(x, y, deg, rcond=None, full=False):
    'units wrapped numpy.polyfit'
    
    _polyfit = np.polyfit

    if full:
        dP, residuals, rank, singular_values, rcond  = np.polyfit(np.array(x), np.array(y), deg, rcond, full)
    else:
        dP = np.polyfit(np.array(x), np.array(y), deg, rcond, full)

    # now we need to put units on P
    # p(x) = p[0] * x**deg + ... + p[deg]
    P = []
    for i, p in enumerate(dP):
        power = deg - i
        X = x**power

        # units on ith element of P from y / X
        uX = Unit(1.0, X.exponents, X.label)
        uy = Unit(1.0, y.exponents, y.label)
        uPi = uy / uX
        # so annoying. if you do dP[i] * uP you lose units.
        P += [uPi * p]

    if full:
        return P, residuals, rank, singular_values, rcond
    else:
        return P

def polyder(P, m=1):
    'units wrapped numpy.polyder'

    dP = np.array(P)

    # loop for as many derivatives as we request
    for j in range(0, m):
        # first, get values of the derivative
        dP = np.polyder(np.array(dP))

        # container to hold values with units.
        # p(x) = p[0] * x**deg + ... + p[deg]
        # dp(x) = deg * p[0] * x**(deg - 1)+ ... + p[-2]
        # so the units on dp[i] = units on p[i + 1]/
        udP = []
        for i in range(0, len(dP)):
            # this has units and value=1
            uP = Unit(1.0, P[i].exponents, P[i].label)
            # so annoying. if you do dP[i] * uP you lose units.
            udP += [uP * dP[i]]
            
    return udP  
        

from scipy.integrate import quad as _quad

def quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08,
         epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None,
         wopts=None, maxp1=50, limlst=50):
    'units wrapped scipy.integrate.quad'

    def wrappedfunc(x, *args):
        return float(func(x, *args))

    # get the units on the integral
    INTEGRAND = func(a, *args)
    U = INTEGRAND*a

    if full_output:
        (y, abserr,
         infodict,
         message, explain) = _quad(wrappedfunc, float(a), float(b), args,
                                   full_output, epsabs, epsrel, limit, points,
                                   weight, wvar, wopts, maxp1, limlst)
        return (Unit(y, U.exponents, U.label),
                Unit(abserr, U.exponents, U.label),
                infodict, message, explain)
    else:
        (y, abserr) = _quad(wrappedfunc, float(a), float(b), args,
                                   full_output, epsabs, epsrel, limit, points,
                                   weight, wvar, wopts, maxp1, limlst)
        
        return (Unit(y, U.exponents, U.label),
                Unit(abserr, U.exponents, U.label))
