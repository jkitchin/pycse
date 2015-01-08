import numpy as np
from pycse.units import *
from nose.tools import raises


@raises(IncompatibleUnits)
def test1():
    '''float64 comparison

    there is a function call order resolution here. when you do
    float64 == unit, the __eq__ of the float64 is called, and this
    returns true sometimes. it should always return false.

    unit == float64 works as it should
    '''
    u = units()

    a = np.float64(2)
    # should raise an exception
    print '{0} float 64 == {1} unit : {2}'.format(a, 2 * u.m, a == 2 * u.m)


@raises(IncompatibleUnits)
def test2():
    '''float64 comparison

    there is a function call order resolution here. when you do
    float64 == unit, the __eq__ of the float64 is called, and this
    returns true sometimes. it should always return false.

    unit == float64 works as it should
    '''
    u = units()

    a = np.float64(2)
    2 * u.m == a  # should raise exception

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
