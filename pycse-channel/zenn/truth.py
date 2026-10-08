"""Ground-truth zentropy model used in the ZENN video example.

Three configurations of an order parameter x:
  L : ordered, well at x = -1
  R : ordered, well at x = +1
  D : disordered, well at x = 0, higher energy but higher entropy
F(x, T) = -kT ln sum_k exp(-(E_k - T S_k) / kT)
"""

import numpy as np

A = 1.0  # well stiffness
DE = 2.0  # energy penalty of the disordered configuration
DS = 3.0  # entropy gain of the disordered configuration


def configs(x):
    E = np.stack([A * (x + 1) ** 2, A * (x - 1) ** 2, A * x**2 + DE], axis=-1)
    S = np.stack([0 * x, 0 * x, 0 * x + DS], axis=-1)
    return E, S


def F_true(x, T):
    E, S = configs(x)
    T = np.broadcast_to(T, x.shape)[..., None]
    Fk = E - T * S
    m = (-Fk / T).max(axis=-1, keepdims=True)
    return -T[..., 0] * (m[..., 0] + np.log(np.exp(-Fk / T - m).sum(axis=-1)))


def p_true(x, T):
    E, S = configs(x)
    Fk = E - T * S
    w = np.exp(-(Fk - Fk.min(axis=-1, keepdims=True)) / T)
    return w / w.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    from scipy.optimize import brentq

    def d2F0(T, h=1e-3):
        x = np.array([-h, 0.0, h])
        f = F_true(x, T)
        return (f[0] - 2 * f[1] + f[2]) / h**2

    for T in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        print(T, d2F0(T))
    print("Tc =", brentq(d2F0, 0.3, 1.0))
