from pycse.units import *

def test1():
    'test 1d fsolve'
    u = units()

    def func(x):
        return x**2 - 4*u.kg**2

    from pycse.umath import fsolve

    X = fsolve(func, 1.5*u.kg)

    assert X == 2.0 * u.kg
