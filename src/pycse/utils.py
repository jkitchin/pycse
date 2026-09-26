"""Provides utility functions in pycse.

1. Fuzzy comparisons for float numbers.
2. An ignore exception context manager
3. A handy function to read a google sheet.
"""

# Copyright 2015, John Kitchin
# (see accompanying license files for details).
import re
from urllib.parse import urlparse
from contextlib import contextmanager
import numpy as np
import pandas as pd


def _tol(x, y, epsilon, rtol):
    """Return the comparison tolerance epsilon + rtol * max(|x|, |y|)."""
    if rtol:
        return epsilon + rtol * np.maximum(np.abs(x), np.abs(y))
    return epsilon


def _result(r):
    """Return a Python bool for scalar results, else a boolean array."""
    if np.ndim(r) == 0:
        return bool(r)
    return r


def feq(x, y, epsilon=np.spacing(1), rtol=0):
    """Fuzzy equals.

    x == y with tolerance epsilon + rtol * max(|x|, |y|).

    epsilon is an absolute tolerance, rtol is an optional relative tolerance.
    Works elementwise on arrays; returns a bool for scalar inputs.
    """
    tol = _tol(x, y, epsilon, rtol)
    return _result(np.logical_not(np.logical_or(x < (y - tol), y < (x - tol))))


def flt(x, y, epsilon=np.spacing(1), rtol=0):
    """Fuzzy less than.

    x < y with tolerance epsilon + rtol * max(|x|, |y|).
    Works elementwise on arrays; returns a bool for scalar inputs.
    """
    return _result(x < (y - _tol(x, y, epsilon, rtol)))


def fgt(x, y, epsilon=np.spacing(1), rtol=0):
    """Fuzzy greater than.

    x > y with tolerance epsilon + rtol * max(|x|, |y|).
    Works elementwise on arrays; returns a bool for scalar inputs.
    """
    return _result(y < (x - _tol(x, y, epsilon, rtol)))


def fle(x, y, epsilon=np.spacing(1), rtol=0):
    """Fuzzy less than or equal to.

    x <= y with tolerance epsilon + rtol * max(|x|, |y|).
    Works elementwise on arrays; returns a bool for scalar inputs.
    """
    return _result(np.logical_not(y < (x - _tol(x, y, epsilon, rtol))))


def fge(x, y, epsilon=np.spacing(1), rtol=0):
    """Fuzzy greater than or equal to.

    x >= y with tolerance epsilon + rtol * max(|x|, |y|).
    Works elementwise on arrays; returns a bool for scalar inputs.
    """
    return _result(np.logical_not(x < (y - _tol(x, y, epsilon, rtol))))


@contextmanager
def ignore_exception(*exceptions):
    """Context manager to ignore EXCEPTIONS raised in its body.

    A message is printed when an exception is caught.

    >>> with ignore_exception(ZeroDivisionError):
    ...     print(1/0)
    caught division by zero

    """
    try:
        yield
    except exceptions as e:
        print("caught {}".format(e))


def read_gsheet(url, *args, **kwargs):
    """Return a dataframe for the Google Sheet at url.

    args and kwargs are passed to pd.read_csv
    The url should be viewable by anyone with the link.
    """
    u = urlparse(url)
    if not ((u.netloc == "docs.google.com") and u.path.startswith("/spreadsheets/d/")):
        raise Exception(f"{url} does not seem to be for a sheet")

    fid = u.path.split("/")[3]
    result = re.search("gid=([0-9]*)", u.fragment)
    if result:
        gid = result.group(1)
    else:
        # default to main sheet
        gid = 0

    purl = f"https://docs.google.com/spreadsheets/d/{fid}/export?format=csv&gid={gid}"

    return pd.read_csv(purl, *args, **kwargs)
