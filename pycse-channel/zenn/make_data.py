"""Bundle results.json + the true configurations into widgets/data.js.

A .js file (not .json) so the widgets work from file:// without a server.
"""

import json
import numpy as np
from truth import configs, p_true, F_true

d = json.load(open("results.json"))
x = np.array(d["x"])
T = np.array(d["T"])


def r(a, n=4):
    return np.round(np.asarray(a, dtype=float), n).tolist()


E, S = configs(x)
truth = {
    "Fk": [r((E - t * S).T) for t in T],  # (nT, 3, nx)
    "p": [r(p_true(x, t).T) for t in T],
}

# Fine T grid for the zentropy explainer (smooth slider)
Tf = np.round(np.arange(0.1, 2.0001, 0.01), 3)
fine = {
    "T": Tf.tolist(),
    "F": [r(F_true(x, t)) for t in Tf],
    "Fk": [r((E - t * S).T) for t in Tf],
    "p": [r(p_true(x, t).T) for t in Tf],
}

out = {
    "x": r(x), "T": r(T), "T_train": d["T_train"], "Tc": d["Tc"],
    "summary": d["summary"], "train": {k: r(v) for k, v in d["train"].items()},
    "F": {k: [r(v) for v in vs] for k, vs in d["F"].items()},
    "d2F0": {k: r(v, 3) for k, v in d["d2F0"].items()},
    "learned": {"p": [r(v, 3) for v in d["learned"]["p"]],
                "Fk": [r(v) for v in d["learned"]["Fk"]]},
    "truth": truth, "fine": fine,
}
with open("widgets/data.js", "w") as fh:
    fh.write("window.ZD = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("wrote widgets/data.js")
