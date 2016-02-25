from pycse import odelay
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ode(y, x, k):
    return -k * y


def event(y, x):
    value = y - 0.3
    isterminal = True
    direction = 0
    return value, isterminal, direction

xspan = np.linspace(0, 3)

y0 = 1


for k in [2, 3, 4]:
    X, Y, XE, YE, IE = odelay(ode, y0, xspan, events=[event], args=(k,))
    plt.plot(X, Y)


# plt.show()
