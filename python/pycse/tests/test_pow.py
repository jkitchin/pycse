from pycse.units import *
from nose.tools import raises

@raises(IncompatibleUnits)
def test1():
    'test unit in pow'
    m = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')              
    print 1**m

def test2():
    'test pow'
    m = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')              
    assert m**2 == Unit(1.0, [2,0,0,0,0,0,0,0], 'm')

def test3():
    'test ipow'
    m = Unit(1.0, [1,0,0,0,0,0,0,0], 'm')
    m **= 2
    assert m == Unit(1.0, [2,0,0,0,0,0,0,0], 'm')
