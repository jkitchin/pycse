"""Beginner module.

This module contains function definitions designed to minimize the need for
Python syntax. They are for beginners just learning, and they try to give
helpful error messages.

There are a series of functions to access parts of a list, e.g. the first-fifth
and nth elements, the last element, all but the first, and all but the last
elements. There is also a cut function to avoid list slicing syntax.

"""

import collections

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
from scipy.optimize import fsolve as _fsolve

def fsolve(objective, x0, *args, **kwargs):
    """A Wrapped version of scipy.optimize.fsolve.

    objective: a callable function f(x) = 0
    x0: the initial guess for the solution.

    This version warns you if the call did not finish cleanly and prints the message.

    Returns:
       If there is only one result it returns a float, otherwise it returns an array.
    """
    ans, info, flag, msg = _fsolve(objective, x0, full_output=1, *args, **kwargs)

    if flag != 1:
        raise Exception('fsolve did not finish cleanly: {}'.format(msg))

    if len(ans) == 1:
        return float(ans)
    else:
        return ans
