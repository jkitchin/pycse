from pycse import odelay, feq
import numpy as np

def ode(y, x, k):
    return k

def event(y, x):
    value = y - 0.3
    isterminal = True
    direction = 0
    return value, isterminal, direction

xspan = np.linspace(0, 3)

y0 = 0

k = 1

def test_odelay_event():
    X, Y, XE, YE, IE = odelay(ode, y0, xspan, events=[event], args=(k,))
    assert feq(XE[0], 0.3)
