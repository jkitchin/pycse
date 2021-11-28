# Copyright 2015, John Kitchin
# (see accompanying license files for details).
import numpy as np
from contextlib import contextmanager
from urllib.parse import urlparse
import pandas as pd
import re


def feq(x, y, epsilon=np.spacing(1)):
    """x == y with tolerance"""
    return not((x < (y - epsilon)) or (y < (x - epsilon)))


def flt(x, y, epsilon=np.spacing(1)):
    'x < y with tolerance'
    return x < (y - epsilon)


def fgt(x, y, epsilon=np.spacing(1)):
    'x > y with tolerance'
    return y < (x - epsilon)


def fle(x, y, epsilon=np.spacing(1)):
    'x <= y with tolerance'
    return not(y < (x - epsilon))


def fge(x, y, epsilon=np.spacing(1)):
    'x >= y with tolerance'
    return not(x < (y - epsilon))


@contextmanager
def ignore_exception(*exceptions):
    """Decorator to ignore exceptions.

    >>> with ignore_exception(ZeroDivisionError):
    ...     print(1/0)

    """
    try:
        yield
    except exceptions as e:
        print('caught {}'.format(e))
        return
    finally:
        print('done')


def read_gsheet(url, *args, **kwargs):
    '''Return a dataframe for the Google Sheet at url.
    args and kwargs are passed to pd.read_csv

    The url should be viewable by anyone with the link.'''

    u = urlparse(url)
    if not (u.netloc == 'docs.google.com') and u.path.startswith('/spreadsheets/d/'):
        raise Exception(f'{url} does not seem to be for a sheet')
    
    fid = u.path.split('/')[3]
    result = re.search('gid=([0-9]*)', u.fragment)
    if result:
        gid = result.group(1)
    else:
        # default to main sheet
        gid = 0
    
    purl = (f'https://docs.google.com/spreadsheets/d/{fid}/export?format=csv&gid={gid}')

    return pd.read_csv(purl, *args, **kwargs)
        
