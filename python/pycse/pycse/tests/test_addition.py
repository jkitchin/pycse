from pycse.units import *
from nose.tools import raises


@raises(IncompatibleUnits)
def test1():
    'test + constant'
    m = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')              
    print 1 + m

@raises(IncompatibleUnits)
def test2():
    'test + constant'
    m = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')              
    print m + 1

@raises(IncompatibleUnits)
def test3():
    'test iadd with constant'
    m = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')
    m += 1

@raises(IncompatibleUnits)
def test5():
    'test incomp units'
    m = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')
    kg = Unit(1.0, [0,1,0,0,0,0,0,0], 'm')
    return m + kg

def test4():
    'test iadd with unit'
    m = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')
    m += m
    assert m == Unit(2.0, [1,0,0,0,0,0,0,0], 'm')  

def test6():
    'add 2 units'
    m1 = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')
    m2 = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')
    assert (m1 + m2) == Unit(2.0, [1,0,0,0,0,0,0,0], 'm')  

def test7():
    'add unit arrays'
    m = Unit([0, 1, 2], [1,0,0,0,0,0,0,0], 'm')
    assert (m + m) == Unit([0, 2, 4], [1,0,0,0,0,0,0,0], 'm')

def test8():
    'add 2d unit arrays'
    a = np.array([[0, 1],[2, 3]])
    m = Unit(a, [1,0,0,0,0,0,0,0], 'm')
    assert (m + m) == Unit([[0, 2],[4, 6]], [1,0,0,0,0,0,0,0], 'm')
