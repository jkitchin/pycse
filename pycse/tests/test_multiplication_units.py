from pycse.units import *
from nose.tools import raises


def test1():
    m = Unit(1.0, [1, 0, 0, 0, 0, 0, 0], 'm')
    kg = Unit(1.0, [0, 1, 0, 0, 0, 0, 0], 'kg')

    p = m * kg

    assert p == Unit(1.0, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test2():
    'Test if a product becomes dimensionless'
    m1 = Unit(1.0, [1, 0, 0, 0, 0, 0, 0], 'm')
    m2 = Unit(1.0, [-1, 0, 0, 0, 0, 0, 0], '1/m')

    p = m1 * m2

    assert p == 1

def test3():
    'test constant multiplication'
    m = Unit(1.0, [1, 0, 0, 0, 0, 0, 0], 'm')

    assert (3 * m) == Unit(3.0, [1, 0, 0, 0, 0, 0, 0], 'm')
    assert (m * 3) == Unit(3.0, [1, 0, 0, 0, 0, 0, 0], 'm')


def test4():
    'test imul with constant'
    m = Unit(1.0, [1, 0, 0, 0, 0, 0, 0], 'm')
    m *= 2
    assert m == Unit(2.0, [1, 0, 0, 0, 0, 0, 0], 'm')


def test5():
    'test imul with unit'
    m = Unit(1.0, [1, 0, 0, 0, 0, 0, 0], 'm')
    kg = Unit(1.0, [0, 1, 0, 0, 0, 0, 0], 'kg')

    m *= kg
    assert m == Unit(1.0, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test6():
    'test element-wise *'
    m = Unit([1.0, 2, 3], [1, 0, 0, 0, 0, 0, 0], 'm')

    assert (m * m) == Unit([1, 4, 9], [2, 0, 0, 0, 0, 0, 0], 'm')


def test7():
    'test np.dot'
    m = Unit([1.0, 2, 3], [1, 0, 0, 0, 0, 0, 0], 'm')
    assert np.dot(m, m) == Unit(14, [2, 0, 0, 0, 0, 0, 0], 'm')
