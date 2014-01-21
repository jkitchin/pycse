import numpy as np

def feq(x, y, epsilon=np.spacing(1)):
    'x == y'
    return not((x < (y - epsilon)) or (y < (x - epsilon)))

def flt(x, y, epsilon=np.spacing(1)):
    'x < y'
    return x < (y - epsilon)

def fgt(x, y, epsilon=np.spacing(1)):
    'x > y'
    return y < (x - epsilon)

def fle(x, y, epsilon=np.spacing(1)):
    'x <= y'
    return not(y < (x - epsilon))

def fge(x, y, epsilon=np.spacing(1)):
    'x >= y'
    return not(x < (y - epsilon))
    
    
from contextlib import contextmanager
@contextmanager
def ignore_exception(*exceptions):
    try:
        yield
    except exceptions as e:
        print 'caught ',e
        return
    finally:
        print 'done'

with ignore_exception(ZeroDivisionError):
    print 1/0