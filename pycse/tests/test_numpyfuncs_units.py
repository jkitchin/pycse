import numpy as np
from pycse.units import *
from nose.tools import raises
from pycse.utils import *

def test1():
    'test trapz'
    u = units()
    v = np.ones(10,dtype=np.float) * u.m / u.s
    t = np.linspace(0, 10, 10) * u.s

    assert np.trapz(v, t) == 10*u.m

def test2():
    'min, max'
    u = units()
    a = np.linspace(-1, 1) * u.m
    assert min(a) == -1*u.m
    assert a.min() == -1*u.m
    assert max(a) == 1*u.m
    assert a.max() == 1*u.m
    assert np.amin(a) == -1*u.m
    assert np.amax(a) == 1*u.m

@raises(IncompatibleUnits)
def test3():
    'test polyfit 1'
    u = units()
    t = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]) # no dimension
    Cc = 2 * u.mol / u.m**3 * t

    P = np.polyfit(t, Cc, 1) # these do not have dimensions
    
    assert (2 * u.mol / u.m**3) == P[0]
    assert 0 * u.mol / u.m**3 == P[1]
    

def test_polyfit1():
    u = units()
    from pycse.umath import polyfit
    

    # raw data provided
    t = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0] * u.s;
    Cc = 2*u.mol / u.m**3 / u.s * t

    P = polyfit(t,Cc,1)
    
    assert np.array(np.abs(P[0] - 2 * u.mol / u.m**3 / u.s)) <= 1e-6
    assert np.array(np.abs(P[1] == 0 * u.mol / u.m**3)) <= 1e-6

test_polyfit1()

def test_polyfit2():
    u = units()
    from pycse.umath import polyfit
    t = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0] * u.dimensionless;
    Cc = 2*u.mol / u.m**3 * t

    P = polyfit(t,Cc,1);
    assert P[0] == 2 * u.mol / u.m**3 
    assert P[1] == 0 * u.mol / u.m**3
    
def test_polyder():
    
    u = units()
    from pycse.umath import polyfit, polyder
    t = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0] * u.s;
    Cc = 2.0 * u.mol / u.m**3 / u.s * t

    P = polyfit(t, Cc, 1)
    
    dP = polyder(P, 1)
    
    assert np.abs(float(dP[0] - 2.0 * u.mol / u.m**3 / u.s)) < 1e-6
    
def test_polyder_2():
    u = units()
    from pycse.umath import polyfit, polyder
    x = np.linspace(0.1, 2) * u.s;

    a = 25 * u.m / u.s**2
    v = 5 * u.m / u.s
    b = 1 * u.m

    y = a * x**2 + v * x + b
    
    P = polyfit(x, y, 2)
    
    dP = polyder(P, 1)

    assert np.abs(float(dP[0] - (2 * a))) < 1e-6
    assert np.abs(float(dP[1] - (v))) < 1e-6

    dP2 = polyder(P, 2)
    assert np.abs(float(dP2[0] - (2 * a))) < 1e-6

