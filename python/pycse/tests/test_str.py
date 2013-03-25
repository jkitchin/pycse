from pycse.units import *
from nose.tools import raises

def test1():
    u = units('SI')
    assert str(u.kJ) == '1000.0 * J'

