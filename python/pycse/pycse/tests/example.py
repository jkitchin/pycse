from pycse.units import *
from pycse.umath import fsolve
import matplotlib.pyplot as plt

u = units()

Cao = 2 * u.mol / u.L
V = 10 * u.L
nu = 0.5 * u.L / u.s
k = 0.23 * u.L / u.mol / u.s

def func(Ca):
    return V - nu * (Cao - Ca) / (k * Ca**2)

Ca = np.linspace(0.001 , 2) * u.mol / u.L
plt.plot(Ca / (u.mol/u.L), func(Ca))
plt.ylim([-0.1, 0.1])
plt.xlabel('$C_A$')
plt.ylabel('$f(C_A)$')
plt.show()

cguess = 0.5 * u.mol / u.L

C = fsolve(func, cguess)
print C
