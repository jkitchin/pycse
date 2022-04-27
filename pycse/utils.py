"""Provides utility functions in pycse.

1. Fuzzy comparisons for float numbers.
2. An ignore exception decorator
3. A handy function to read a google sheet.
"""
# Copyright 2015, John Kitchin
# (see accompanying license files for details).
import re
from urllib.parse import urlparse
from contextlib import contextmanager
import numpy as np
import pandas as pd


def feq(x, y, epsilon=np.spacing(1)):
    """Fuzzy equals.

    x == y with tolerance
    """
    return not ((x < (y - epsilon)) or (y < (x - epsilon)))


def flt(x, y, epsilon=np.spacing(1)):
    """Fuzzy less than.

    x < y with tolerance
    """
    return x < (y - epsilon)


def fgt(x, y, epsilon=np.spacing(1)):
    """Fuzzy greater than.

    x > y with tolerance
    """
    return y < (x - epsilon)


def fle(x, y, epsilon=np.spacing(1)):
    """Fuzzy less than or equal to.

    x <= y with tolerance
    """
    return not (y < (x - epsilon))


def fge(x, y, epsilon=np.spacing(1)):
    """Fuzzy greater than or equal to .

    x >= y with tolerance
    """
    return not (x < (y - epsilon))


@contextmanager
def ignore_exception(*exceptions):
    """Ignore exceptions on decorated function.

    >>> with ignore_exception(ZeroDivisionError):
    ...     print(1/0)

    """
    try:
        yield
    except exceptions as e:
        print("caught {}".format(e))
        return
    finally:
        print("done")


def read_gsheet(url, *args, **kwargs):
    """Return a dataframe for the Google Sheet at url.

    args and kwargs are passed to pd.read_csv
    The url should be viewable by anyone with the link.
    """
    u = urlparse(url)
    if not (u.netloc == "docs.google.com") and u.path.startswith(
        "/spreadsheets/d/"
    ):
        raise Exception(f"{url} does not seem to be for a sheet")

    fid = u.path.split("/")[3]
    result = re.search("gid=([0-9]*)", u.fragment)
    if result:
        gid = result.group(1)
    else:
        # default to main sheet
        gid = 0

    purl = (
        "https://docs.google.com/spreadsheets/d/"
        f"{fid}/export?format=csv&gid={gid}"
    )

    return pd.read_csv(purl, *args, **kwargs)
