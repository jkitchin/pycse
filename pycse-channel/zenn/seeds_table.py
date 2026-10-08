"""Summarize experiment.py over seeds: python experiment.py N for N in 1..3 first."""

import json
import numpy as np

files = ["results.json"] + [f"results_seed{n}.json" for n in (1, 2, 3)]
rows = []
for f in files:
    d = json.load(open(f))
    T = np.array(d["T"])
    cold = [i for i, t in enumerate(T) if t <= 0.25 + 1e-9]
    c = {k: np.mean(np.abs(np.array(d["d2F0"][k])[cold] - np.array(d["d2F0"]["true"])[cold]))
         for k in ("zenn", "mlp")}
    s = d["summary"]
    rows.append([s["interp (0.55-0.75)"]["zenn"], s["interp (0.55-0.75)"]["mlp"],
                 s["extrap cold (0.15-0.25)"]["zenn"], s["extrap cold (0.15-0.25)"]["mlp"],
                 s["extrap hot (1.2-2.0)"]["zenn"], s["extrap hot (1.2-2.0)"]["mlp"],
                 c["zenn"], c["mlp"],
                 abs(d["Tc"]["zenn"] - d["Tc"]["true"]), abs(d["Tc"]["mlp"] - d["Tc"]["true"])])
rows = np.array(rows)
names = ["interp RMSE", "cold RMSE", "hot RMSE", "cold |F''(0) err|", "|Tc err|"]
print(f"{'':20s} {'ZENN mean':>10s} {'NN mean':>10s}  ZENN wins")
for j, n in enumerate(names):
    z, m = rows[:, 2 * j], rows[:, 2 * j + 1]
    print(f"{n:20s} {z.mean():10.4f} {m.mean():10.4f}  {int((z < m).sum())}/4   z={np.round(z,3)} m={np.round(m,3)}")
