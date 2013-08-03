from pycse.units import *
from nose.tools import raises


def test1():      
    m = Unit(1.0, [1,0,0,0,0,0,0], 'm')
    kg = Unit(1.0, [0,1,0,0,0,0,0], 'kg')

    p = m / kg

    assert p == Unit(1.0, [1.0, -1.0,0.0,0.0,0.0,0.0,0.0])


def test2():
    'Test if a division becomes dimensionless'
    m1 = Unit(1.0, [1,0,0,0,0,0,0], 'm')
    
    p = m1 / m1

    assert p == 1

def test3():
    'test constant division'
    m = Unit(1.0, [1,0,0,0,0,0,0], 'm')

    assert (3 / m) == Unit(3.0, [-1,0,0,0,0,0,0], 'm')
    
    a = m / 3

    assert np.abs(np.array(a) - 1.0/3.0) < 1e-6

    # float tolerance means 1 / 3 != 1.0 / 3.0
    #assert (m / 3) == Unit(1.0/3.0, [1,0,0,0,0,0,0], 'm')


def test4():
    'test idiv with constant'
    m = Unit(1.0, [1,0,0,0,0,0,0], 'm')
    m /= 2
    assert m == Unit(0.5, [1,0,0,0,0,0,0], 'm')

def test5():
    'test idiv with unit'
    m = Unit(1.0, [1,0,0,0,0,0,0], 'm')
    kg = Unit(1.0, [0,1,0,0,0,0,0], 'kg')

    m /= kg
    assert m == Unit(1.0, [1.0, -1.0,0.0,0.0,0.0,0.0,0.0])


def test6():
    'test element-wise /'
    m = Unit([1.0, 2, 3], [1,0,0,0,0,0,0], 'm')
    kg = Unit(0.5, [0,1,0,0,0,0,0], 'kg')

    print (m / kg)
    assert (m / kg) == Unit([2, 4, 6], [1, -1,0,0,0,0,0], 'm')


