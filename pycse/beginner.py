"""Beginner module.

This module contains function definitions designed to minimize the need for
Python syntax. They are for beginners just learning, and they try to give
helpful error messages.

There are a series of functions to access parts of a list, e.g. the first-fifth
and nth elements, the last element, all but the first, and all but the last
elements. There is also a cut function to avoid list slicing syntax. The point
of these is to delay introducing indexing syntax.

"""
import numpy as np
import collections
from scipy.optimize import fsolve as _fsolve
from scipy.integrate import quad


def first(x):
    """Return the first element of x if it is iterable, else return x."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    return x[0]


def second(x):
    """Return the second element of x."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    if not len(x) >= 2:
        raise Exception('{} does not have a second element.'.format(x))

    return x[1]


def third(x):
    """Return the third element of x."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    if not len(x) >= 3:
        raise Exception('{} does not have a third element.'.format(x))

    return x[2]


def fourth(x):
    """Return the fourth element of x."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    if not len(x) >= 4:
        raise Exception('{} does not have a fourth element.'.format(x))

    return x[3]


def fifth(x):
    """Return the fifth element of x."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    if not len(x) >= 5:
        raise Exception('{} does not have a fifth element.'.format(x))

    return x[4]


def nth(x, n=0):
    """Return the nth value of x."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    if not len(x) >= n:
        raise Exception('{} does not have an n={} element.'.format(x, n))

    return x[n]


def cut(x, start=0, stop=None, step=None):
    """Alias for x[start:stop:step]

    This is to avoid having to introduce the slicing syntax.
    """
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    return x[slice(start, stop, step)]


def last(x):
    """Return the last element of x if it is iterable."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    return x[-1]


def rest(x):
    """Return everything after the first element of x."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    return x[1:]


def butlast(x):
    """Return everything but the last element of x."""
    if not isinstance(x, collections.Iterable):
        raise Exception('{} is not iterable.'.format(x))

    return x[0:-1]


# * Wrapped functions

# These functions are wrapped to provide a simpler use for new students. Usually
# that means there are fewer confusing outputs. For example fsolve returns an
# array even for a single number which leads to the need to unpack it to get a
# simple number. The nsolve function does not do that. It returns a float if the
# result is a 1d array. It also is more explicit about checking for convergence.


def nsolve(objective, x0, *args, **kwargs):
    """A Wrapped version of scipy.optimize.fsolve.

    objective: a callable function f(x) = 0
    x0: the initial guess for the solution.

    This version warns you if the call did not finish cleanly and prints the message.

    Returns:
       If there is only one result it returns a float, otherwise it returns an array.
    """
    if 'full_output' not in kwargs:
        kwargs['full_output'] = 1

    ans, info, flag, msg = _fsolve(objective, x0, *args, **kwargs)

    if flag != 1:
        raise Exception('nsolve did not finish cleanly: {}'.format(msg))

    if len(ans) == 1:
        return float(ans)
    else:
        return ans

# The quad function returns the integral and error estimate. We rarely use the
# error estimate, so here we eliminate it from the output.


def integrate(f, a, b, *args, **kwargs):
    """Integrate the function f(x) from a to b.

    This wraps scipy.integrate.quad to eliminate the error estimate and provide
    better debugging information.

    If the error estimate is greater than the tolerance argument, an exception
    is raised.

    """
    if 'full_output' not in kwargs:
        kwargs['full_output'] = 1
    results = quad(f, a, b, *args, **kwargs)

    tolerance = kwargs.get('tolerance', 1e-6)

    if second(results) > tolerance:
        raise Exception('Your integral error {} is too large. '.format(err)
                        + '{} '.format(fourth(results))
                        + 'See your instructor for help')
    return first(results)


def heaviside(x):
    """Return the heaviside function of x.
    This is equal to zero for x < 0, 0.5 for x==0, and 1 for x > 0.
    """
    return 0.5 * (np.sign(x) + 1)


def step(x):
    """Alias for heaviside(x)."""
    return heaviside(x)
