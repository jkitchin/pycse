from pycse.units import *
from pycse.umath import odeint

def test1():
    'odeint'
    u = units()
    tspan = np.linspace(0, 1)*u.s
    x0 = 0.0 * u.m

    v = 3 * u.m / u.s

    def dxdt(x, t):
        return v

    sol = odeint(dxdt, x0, tspan)
    print sol.shape
    print sol[-1]
    assert sol[-1] == 3 * u.m
test1()
