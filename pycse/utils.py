import numpy as np

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
    
    
from contextlib import contextmanager
@contextmanager
def ignore_exception(*exceptions):
    """Decorator to ignore exceptions.

    >>> with ignore_exception(ZeroDivisionError):
    ...     print 1/0

    """
    try:
        yield
    except exceptions as e:
        print 'caught ',e
        return
    finally:
        print 'done'

if __name__ == '__main__':
    with ignore_exception(ZeroDivisionError):
        print 1/0
