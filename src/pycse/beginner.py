"""Beginner module.

This module contains function definitions designed to minimize the need for
Python syntax. They are for beginners just learning, and they try to give
helpful error messages.

There are a series of functions to access parts of a list, e.g. the first-fifth
and nth elements, the last element, all but the first, and all but the last
elements. There is also a cut function to avoid list slicing syntax. The point
of these is to delay introducing indexing syntax.

"""

import collections.abc
from scipy.optimize import fsolve as _fsolve
from scipy.integrate import quad


class NsolveError(RuntimeError):
    """Raised when nsolve does not finish cleanly."""


class IntegrateError(RuntimeError):
    """Raised when the integrate error estimate is too large."""


def first(x):
    """Return the first element of x if it is iterable, else return x."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    return x[0]


def second(x):
    """Return the second element of x."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    if not len(x) >= 2:
        raise Exception("{} does not have a second element.".format(x))

    return x[1]


def third(x):
    """Return the third element of x."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    if not len(x) >= 3:
        raise Exception("{} does not have a third element.".format(x))

    return x[2]


def fourth(x):
    """Return the fourth element of x."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    if not len(x) >= 4:
        raise Exception("{} does not have a fourth element.".format(x))

    return x[3]


def fifth(x):
    """Return the fifth element of x."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    if not len(x) >= 5:
        raise Exception("{} does not have a fifth element.".format(x))

    return x[4]


def nth(x, n=0):
    """Return the nth value of x.
    Note that `n` starts at 0."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    if not len(x) >= n:
        raise Exception("{} does not have an n={} element.".format(x, n))

    return x[n]


def cut(x, start=0, stop=None, step=None):
    """Alias for x[start:stop:step].

    This is to avoid having to introduce the slicing syntax.
    """
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    return x[slice(start, stop, step)]


def last(x):
    """Return the last element of x if it is iterable."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    return x[-1]


def rest(x):
    """Return everything after the first element of x."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    return x[1:]


def butlast(x):
    """Return everything but the last element of x."""
    if not isinstance(x, collections.abc.Iterable):
        raise Exception("{} is not iterable.".format(x))

    return x[0:-1]


# * Wrapped functions

# These functions are wrapped to provide a simpler use for new students. Usually
# that means there are fewer confusing outputs. For example fsolve returns an
# array even for a single number which leads to the need to unpack it to get a
# simple number. The nsolve function does not do that. It returns a float if the
# result is a 1d array. It also is more explicit about checking for convergence.


def nsolve(objective, x0, *args, **kwargs):
    """Solve an objective function.

    A Wrapped version of scipy.optimize.fsolve.

    objective: a callable function f(x, *args) = 0
    x0: the initial guess for the solution.
    args: extra positional arguments are passed to the objective function.
    kwargs: passed to scipy.optimize.fsolve.

    This version raises an NsolveError (a subclass of RuntimeError) if the call
    did not finish cleanly, and includes the message from fsolve.

    Returns: If there is only one result it returns a float, otherwise it
       returns an array.

    """
    if "full_output" not in kwargs:
        kwargs["full_output"] = 1

    if args:
        # extra positional args are for the objective, not fsolve's positional
        # parameters (args, fprime, ...).
        kwargs["args"] = tuple(args)

    ans, _, flag, msg = _fsolve(objective, x0, **kwargs)

    if flag != 1:
        raise NsolveError("nsolve did not finish cleanly: {}".format(msg))

    if len(ans) == 1:
        # Use item() for NumPy 2.x compatibility (Python 3.13+)
        # This works for both 0-d and 1-d arrays with single element
        return float(ans[0]) if hasattr(ans, "__getitem__") else float(ans)
    else:
        return ans


# The quad function returns the integral and error estimate. We rarely use the
# error estimate, so here we eliminate it from the output.


def integrate(f, a, b, *args, **kwargs):
    """Integrate the function f(x) from a to b.

    This wraps scipy.integrate.quad to eliminate the error estimate and provide
    better debugging information.

    Extra positional args are passed to f, i.e. f(x, *args). Other kwargs are
    passed to scipy.integrate.quad, except for these two:

    tolerance: absolute error tolerance (default 1e-6).
    rtol: relative error tolerance (default 1e-6).

    If the error estimate is greater than max(tolerance, rtol * abs(integral)),
    an IntegrateError (a subclass of RuntimeError) is raised.

    """
    tolerance = kwargs.pop("tolerance", 1e-6)
    rtol = kwargs.pop("rtol", 1e-6)

    if "full_output" not in kwargs:
        kwargs["full_output"] = 1

    if args:
        # extra positional args are for f, not quad's positional parameters.
        kwargs["args"] = tuple(args)

    results = quad(f, a, b, **kwargs)

    value, err = first(results), second(results)

    if err > max(tolerance, rtol * abs(value)):
        # quad only includes a message (4th element) when there was a problem.
        msg = "{} ".format(results[3]) if len(results) > 3 else ""
        raise IntegrateError(
            "Your integral error {} is too large. ".format(err)
            + msg
            + "See your instructor for help"
        )
    return value
