from pycse import odelay

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

Ca0 = 3.0  # mol / L
v0 = 10.0  # L / min
k = 0.23   # 1 / min

Fa_Exit = 0.3 * v0


def ode(Fa, V):
    Ca = Fa / v0
    return -k * Ca


def event1(Fa, V):
    isterminal = False
    direction = 0
    value = Fa - Fa_Exit
    return value, isterminal, direction

Vspan = np.linspace(0, 200)  # L

V, F, TE, YE, IE = odelay(ode, Ca0 * v0, Vspan, events=[event1])

print('Solution is at {0} L'.format(V[-1]))


matplotlib.use('Agg')

plt.plot(V, F)
