"""ZENN vs a regular neural network on a temperature-dependent free energy.

Train on noisy F(x, T) at a handful of temperatures, then ask both models
about temperatures they never saw, the curvature d2F/dx2, and the
critical temperature where the double well merges into a single well.
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import brentq
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pycse.sklearn.zenn import ZENNRegressor
from truth import F_true, p_true

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(42 + SEED)
T_train = [0.3, 0.4, 0.5, 0.8, 1.0]
NOISE = 0.02

x1 = np.linspace(-1.8, 1.8, 25)
X, T, y = [], [], []
for t in T_train:
    X.append(x1)
    T.append(np.full_like(x1, t))
    y.append(F_true(x1, t) + NOISE * rng.standard_normal(x1.size))
X, T, y = map(np.concatenate, (X, T, y))
print(f"{X.size} training points at T = {T_train}")

# ZENN: x is the feature, T goes in as temperature
t0 = time.time()
zenn = ZENNRegressor(
    n_configs=4, hidden_dims=(16, 16), loss_type="mse",
    learning_rate=5e-3, max_epochs=8000, random_state=SEED,
)
zenn.fit(X[:, None], y, T=T)
t_zenn = time.time() - t0

# Regular NN: (x, T) are both just inputs
t0 = time.time()
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs",
                 max_iter=5000, alpha=1e-4, random_state=SEED),
)
mlp.fit(np.column_stack([X, T]), y)
t_mlp = time.time() - t0
print(f"train time: ZENN {t_zenn:.1f} s, MLP {t_mlp:.1f} s")


def f_zenn(x, t):
    return zenn.predict(x[:, None], T=np.full_like(x, t))


def f_mlp(x, t):
    return mlp.predict(np.column_stack([x, np.full_like(x, t)]))


def d2(f, t, x0=0.0, h=0.02):
    v = f(np.array([x0 - h, x0, x0 + h]), t)
    return float((v[0] - 2 * v[1] + v[2]) / h**2)


def Tc(f):
    try:
        return brentq(lambda t: d2(f, t), 0.3, 1.0)
    except ValueError:
        return float("nan")


xg = np.linspace(-1.8, 1.8, 91)
Tg = np.round(np.arange(0.15, 2.001, 0.05), 3)


def rmse(f, t):
    return float(np.sqrt(np.mean((f(xg, t) - F_true(xg, t)) ** 2)))


regions = {
    "train T": T_train,
    "interp (0.55-0.75)": [0.55, 0.6, 0.65, 0.7, 0.75],
    "extrap hot (1.2-2.0)": [1.2, 1.4, 1.6, 1.8, 2.0],
    "extrap cold (0.15-0.25)": [0.15, 0.2, 0.25],
}
summary = {}
for name, ts in regions.items():
    summary[name] = {m: float(np.mean([rmse(f, t) for t in ts]))
                     for m, f in [("zenn", f_zenn), ("mlp", f_mlp)]}
    print(f"{name:26s} RMSE  ZENN {summary[name]['zenn']:.4f}   MLP {summary[name]['mlp']:.4f}")

f_true = lambda x, t: F_true(x, t)
tc = {"true": Tc(f_true), "zenn": Tc(f_zenn), "mlp": Tc(f_mlp)}
print("Tc:", tc)

# What ZENN learned inside: per-configuration F_k(x) and weights p_k(x) at each T
land = [zenn.get_energy_landscape(xg[:, None], T=float(t)) for t in Tg]
learned = {"Fk": [np.asarray(d["F"]).T.tolist() for d in land],
           "p": [np.asarray(d["p"]).T.tolist() for d in land]}

# Data for the interactive slides
out = {
    "x": xg.tolist(), "T": Tg.tolist(), "T_train": T_train, "Tc": tc,
    "noise": NOISE, "summary": summary,
    "train": {"x": X.tolist(), "T": T.tolist(), "y": y.tolist()},
    "F": {"true": [F_true(xg, t).tolist() for t in Tg],
          "zenn": [f_zenn(xg, t).tolist() for t in Tg],
          "mlp": [f_mlp(xg, t).tolist() for t in Tg]},
    "d2F0": {k: [d2(f, t) for t in Tg] for k, f in
             [("true", f_true), ("zenn", f_zenn), ("mlp", f_mlp)]},
    "p_true_x0": [p_true(np.array([0.0]), t)[0].tolist() for t in Tg],
    "learned": learned,
    "loss": zenn.history_["loss"][::40],
    "time": {"zenn": t_zenn, "mlp": t_mlp},
}
fname = "results.json" if SEED == 0 else f"results_seed{SEED}.json"
with open(fname, "w") as fh:
    json.dump(out, fh)
print("wrote", fname)
