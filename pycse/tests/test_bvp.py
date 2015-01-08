from pycse import bvp
import numpy as np

# example from http://200.13.98.241/~martin/irq/tareas1/bvp_paper.pdf


def odefun(Y, x):
    u, v, w, z, y = Y. T
    dudx = 0.5 * u * (w - u) / v
    dvdx = -0.5 * (w - u)
    dwdx = (0.9 - 1000 * (w - y) - 0.5 * w * (w - u)) / z
    dzdx = 0.5 * (w - u)
    dydx = -100.0 * (y - w)
    return np.column_stack([dudx, dvdx, dwdx, dzdx, dydx])


def bcfun(Y):
    # u(0) = v(0) = w(0) = 1, z(0) = -10, w(1) = y(1)
    ua, va, wa, za, ya = Y[0, :]
    ub, vb, wb, zb, yb = Y[-1, :]
    z1 = ua - 1
    z2 = va - 1
    z3 = wa - 1
    z4 = za + 10
    z5 = wb - yb
    return [z1, z2, z3, z4, z5]

x = np.linspace(0, 1)

# initial guess
ux = x**0
vx = x**0
wx = -4.5 * x**2 + 8.91 * x + 1
zx = -10 * x**0
yx = -4.5*x**2 + 9*x + 0.91

Yinit = np.column_stack([ux, vx, wx, zx, yx])

sol = bvp(odefun, bcfun, x, Yinit)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
u = sol[:, 0]
v = sol[:, 1]
w = sol[:, 2]
z = sol[:, 3]
y = sol[:, 4]

plt.plot(x, u, x, v, x, w, x, z + 10, x, y)
plt.legend(['u', 'v', 'w', 'z', 'y'], loc='best')
plt.show()
