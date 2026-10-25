"""Mixed-integer programming (MIP) export of trained piecewise-linear models.

A trained model built from affine maps, ReLUs, piecewise-linear splines and tree
splits is an exactly piecewise-linear function, so it can be embedded in an
optimization problem as mixed-integer linear constraints. This module holds the
solver-side machinery once, and each estimator only describes its trained weights:

- ``Affine``, ``ReLU`` and ``PiecewiseLinear`` are layer specs holding numpy arrays.
- ``MIPNetwork`` is a list of layers plus optional uncertainty (``StdSpec``).
- ``TreeModel`` is a list of ``Leaf`` boxes, each with an affine prediction.
- ``propagate_bounds`` computes interval bounds for every layer, which give the
  big-M constants.
- ``add_to_pyomo`` writes a ``MIPNetwork`` or ``TreeModel`` into a Pyomo block.
- ``MIPExportMixin`` gives an estimator ``to_pyomo()`` from its ``mip_network()``.

The ReLU ``y = max(0, a)`` with bounds ``L <= a <= U`` (``L < 0 < U``) is encoded
with one binary ``d``::

    y >= 0,  y >= a,  y <= a - L (1 - d),  y <= U d

Units that are always active (``L >= 0``) or always inactive (``U <= 0``) need no
binary. Tight input bounds give tight big-M constants and faster solves.

``formulation="convex"`` drops the two upper inequalities, so the model is a pure
LP (no binaries). It is exact for input-convex networks (every weight after the
first ReLU nonnegative) when the output is minimized or bounded above: then
``y >= f(x)`` holds everywhere and ``y = f(x)`` at the optimum.

Uncertainty, when a model has it, has the form ``std = sqrt(||A z + c||^2 + d)``
where ``z`` is the output of some layer (the last-layer features or the ensemble
members). This is a second-order cone, so optimizing e.g. ``mean + k * std`` is a
mixed-integer SOCP that solvers such as Gurobi handle directly.

Example::

    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.T = pyo.Var(bounds=(300, 500))
    m.P = pyo.Var(bounds=(1, 10))
    m.surrogate = pyo.Block()
    model.to_pyomo(block=m.surrogate, inputs=[m.T, m.P])
    m.obj = pyo.Objective(expr=m.surrogate.y[0], sense=pyo.maximize)
    pyo.SolverFactory("highs").solve(m)

Pyomo is an optional dependency (``pip install pycse[mip]``); this module only
imports it when a model is built.
"""

import copy
from dataclasses import dataclass, field

import numpy as np

FORMULATIONS = ("bigm", "convex")


@dataclass
class Affine:
    """Affine layer ``z_out = z_in @ W + x @ Wx + b``.

    Args:
        W: Weight matrix, shape (n_in, n_out).
        b: Bias vector, shape (n_out,).
        Wx: Optional skip connection from the network inputs ``x``, shape
            (n_inputs, n_out), as in input-convex and monotonic networks.
    """

    W: np.ndarray
    b: np.ndarray
    Wx: np.ndarray = None

    def __post_init__(self):
        self.W = np.atleast_2d(np.asarray(self.W, dtype=float))
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        if self.W.shape[1] != self.b.shape[0]:
            raise ValueError(f"W has {self.W.shape[1]} outputs but b has {self.b.shape[0]}.")
        if self.Wx is not None:
            self.Wx = np.atleast_2d(np.asarray(self.Wx, dtype=float))
            if self.Wx.shape[1] != self.n_out:
                raise ValueError(f"Wx has {self.Wx.shape[1]} outputs but W has {self.n_out}.")

    @property
    def n_in(self):
        return self.W.shape[0]

    @property
    def n_out(self):
        return self.W.shape[1]


@dataclass
class ReLU:
    """Elementwise ``max(0, z)``.

    Args:
        concave: Optional boolean mask. Masked units compute ``-max(0, -z) =
            min(0, z)`` instead, the concave counterpart used by monotonic networks.
    """

    concave: np.ndarray = None

    def __post_init__(self):
        if self.concave is not None:
            self.concave = np.asarray(self.concave, dtype=bool).reshape(-1)

    def sign(self, n):
        """+1 for ReLU units and -1 for concave units: ``z_out = s * max(0, s * z)``."""
        if self.concave is None:
            return np.ones(n)
        if len(self.concave) != n:
            raise ValueError(f"concave mask has {len(self.concave)} entries for {n} units.")
        return np.where(self.concave, -1.0, 1.0)


@dataclass
class PiecewiseLinear:
    """Sum of 1D piecewise-linear functions: ``z_out[j] = sum_i f_ij(z_in[i]) + z_in @ W + b``.

    ``f_ij`` interpolates ``values[i][:, j]`` linearly between ``breakpoints[i]`` and
    is constant outside them, as in a KAN layer with linear splines and a clipped
    grid. All functions of one input share its breakpoints, so the MIP needs one set
    of segment binaries per input, not per edge. Inputs may have different numbers
    of breakpoints.

    Args:
        breakpoints: Increasing breakpoints for each input: an (n_in, K) array, or a
            list of n_in 1D arrays.
        values: Function values at the breakpoints: an (n_in, K, n_out) array, or a
            list of (K_i, n_out) arrays.
        W: Optional linear term, shape (n_in, n_out).
        b: Optional bias, shape (n_out,).
    """

    breakpoints: list
    values: list
    W: np.ndarray = None
    b: np.ndarray = None

    def __post_init__(self):
        self.breakpoints = [np.asarray(bp, dtype=float).reshape(-1) for bp in self.breakpoints]
        self.values = [np.atleast_2d(np.asarray(v, dtype=float)) for v in self.values]
        if len(self.values) != len(self.breakpoints):
            raise ValueError("breakpoints and values need one entry per input.")
        n_in, n_out = len(self.breakpoints), self.values[0].shape[1]
        for bp, v in zip(self.breakpoints, self.values):
            if v.shape != (len(bp), n_out):
                raise ValueError(
                    f"values for an input with {len(bp)} breakpoints must be "
                    f"({len(bp)}, {n_out}), got {v.shape}."
                )
            if np.any(np.diff(bp) <= 0):
                raise ValueError("breakpoints must be strictly increasing for each input.")
        self.W = np.zeros((n_in, n_out)) if self.W is None else np.asarray(self.W, dtype=float)
        self.b = np.zeros(n_out) if self.b is None else np.asarray(self.b, dtype=float)
        if self.W.shape != (n_in, n_out) or self.b.shape != (n_out,):
            raise ValueError(f"W must be ({n_in}, {n_out}) and b ({n_out},).")

    @property
    def n_in(self):
        return len(self.breakpoints)

    @property
    def n_out(self):
        return self.values[0].shape[1]

    def segments(self, i, lo, hi):
        """Breakpoints of input i restricted to [lo, hi], and the values there."""
        bp = self.breakpoints[i]
        pts = np.unique(np.concatenate([[lo, hi], bp[(bp > lo) & (bp < hi)]]))
        vals = np.column_stack(
            [np.interp(pts, bp, self.values[i][:, j]) for j in range(self.n_out)]
        )
        return pts, vals

    def __call__(self, z):
        out = z @ self.W + self.b
        for i in range(self.n_in):
            for j in range(self.n_out):
                out[:, j] += np.interp(z[:, i], self.breakpoints[i], self.values[i][:, j])
        return out


@dataclass
class Elementwise1D:
    """Features ``z_out[j] = f_j(z_in[index[j]])`` of one input each.

    Linear features (``funcs[j] is None``) are ``slope[j] * z + intercept[j]`` and are
    exported exactly. Nonlinear features (e.g. the sin/cos of a periodic model) are
    interpolated in the MIP on a grid with spacing at most ``spacing[j]`` over the
    bounds of their input, which ``add_to_pyomo`` knows only when it is called; see
    ``MIPNetwork.encoded``. With ``|f_j''| <= curvature[j]`` the interpolation error of
    feature j is at most ``curvature[j] * h**2 / 8`` for grid spacing ``h``.

    Args:
        n_in: Number of inputs.
        index: Input of each feature, shape (n_out,).
        funcs: Per feature, a vectorized numpy function, or None for a linear feature.
        slope, intercept: Linear features' coefficients (ignored for nonlinear ones).
        spacing: Maximum grid spacing per feature (nonlinear features only).
        curvature: Bound on ``|f''|`` per feature, for ``max_error``.
    """

    n_in: int
    index: np.ndarray
    funcs: list
    slope: np.ndarray = None
    intercept: np.ndarray = None
    spacing: np.ndarray = None
    curvature: np.ndarray = None

    def __post_init__(self):
        self.index = np.asarray(self.index, dtype=int).reshape(-1)
        n = len(self.index)
        if len(self.funcs) != n:
            raise ValueError(f"Got {len(self.funcs)} funcs for {n} features.")

        def vec(v, default):
            return np.full(n, default) if v is None else np.asarray(v, dtype=float).reshape(n)

        self.slope = vec(self.slope, 1.0)
        self.intercept = vec(self.intercept, 0.0)
        self.spacing = vec(self.spacing, np.inf)
        self.curvature = vec(self.curvature, np.nan)
        nonlinear = np.array([f is not None for f in self.funcs])
        if np.any(nonlinear & ~(self.spacing > 0) | nonlinear & ~np.isfinite(self.spacing)):
            raise ValueError("Every nonlinear feature needs a finite positive spacing.")

    @property
    def n_out(self):
        return len(self.index)

    def __call__(self, z):
        out = np.empty((len(z), self.n_out))
        for j, (i, f) in enumerate(zip(self.index, self.funcs)):
            zi = z[:, i]
            out[:, j] = self.slope[j] * zi + self.intercept[j] if f is None else f(zi)
        return out

    def grid(self, i, lo, hi):
        """The interpolation grid of input i on [lo, hi]."""
        js = [j for j in range(self.n_out) if self.index[j] == i and self.funcs[j] is not None]
        if not js or hi <= lo:
            return np.unique([lo, hi])
        h = min(self.spacing[j] for j in js)
        return np.linspace(lo, hi, int(np.ceil((hi - lo) / h)) + 1)

    def to_piecewise(self, lo, hi):
        """The ``PiecewiseLinear`` layer the MIP uses for inputs in [lo, hi]."""
        W = np.zeros((self.n_in, self.n_out))
        b = self.intercept * np.array([f is None for f in self.funcs])
        bps, vals = [], []
        for i in range(self.n_in):
            pts = self.grid(i, lo[i], hi[i])
            if len(pts) == 1:  # a fixed input: widen to keep the breakpoints increasing
                pts = np.array([pts[0], pts[0] + 1.0])
            v = np.zeros((len(pts), self.n_out))
            for j in range(self.n_out):
                if self.index[j] != i:
                    continue
                if self.funcs[j] is None:
                    W[i, j] = self.slope[j]
                else:
                    v[:, j] = self.funcs[j](pts)
            bps.append(pts)
            vals.append(v)
        return PiecewiseLinear(bps, vals, W=W, b=b)

    def max_error(self, lo, hi):
        """Bound on each feature's interpolation error for inputs in [lo, hi]."""
        err = np.zeros(self.n_out)
        for j in range(self.n_out):
            if self.funcs[j] is not None:
                i = self.index[j]
                pts = self.grid(i, lo[i], hi[i])
                h = np.max(np.diff(pts)) if len(pts) > 1 else 0.0
                err[j] = self.curvature[j] * h**2 / 8
        return err


@dataclass
class StdSpec:
    """Predictive standard deviation ``sqrt(||A z + c||^2 + d)``.

    Args:
        tap: Index of the layer whose output is ``z`` (-1 means the inputs).
        A: Matrix, shape (k, len(z)).
        c: Offset, shape (k,). Defaults to zeros.
        d: Constant variance term (>= 0).
    """

    tap: int
    A: np.ndarray
    c: np.ndarray = None
    d: float = 0.0

    def __post_init__(self):
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        if self.c is None:
            self.c = np.zeros(self.A.shape[0])
        self.c = np.asarray(self.c, dtype=float).reshape(-1)
        self.d = float(self.d)
        if self.d < 0:
            raise ValueError(f"d must be nonnegative, got {self.d}.")

    def __call__(self, z):
        """Evaluate the std for a batch ``z`` of shape (n, len(z))."""
        r = np.atleast_2d(z) @ self.A.T + self.c
        return np.sqrt(np.sum(r**2, axis=1) + self.d)


@dataclass
class MIPNetwork:
    """A trained model as a sequence of layers, with optional uncertainty.

    Args:
        layers: List of ``Affine``, ``ReLU`` and ``PiecewiseLinear`` layers; the first
            must be ``Affine`` or ``PiecewiseLinear``.
        std: Optional list of ``StdSpec``, one per output.
    """

    layers: list
    std: list = field(default=None)

    def __post_init__(self):
        first_types = (Affine, PiecewiseLinear, Elementwise1D)
        if not self.layers or not isinstance(self.layers[0], first_types):
            raise ValueError(
                "A MIPNetwork must start with an Affine, PiecewiseLinear or Elementwise1D layer."
            )
        n_x = self.layers[0].n_in
        n = self.layers[0].n_out
        for layer in self.layers[1:]:
            if isinstance(layer, first_types):
                if layer.n_in != n:
                    raise ValueError(f"Layer expects {layer.n_in} inputs but receives {n}.")
                n = layer.n_out
            elif isinstance(layer, ReLU):
                layer.sign(n)  # validates the mask length
            else:
                raise TypeError(f"Unsupported layer type: {type(layer).__name__}")
        for layer in self.layers:
            if isinstance(layer, Affine) and layer.Wx is not None and layer.Wx.shape[0] != n_x:
                raise ValueError(f"Wx has {layer.Wx.shape[0]} rows for {n_x} network inputs.")
        if self.std is not None and len(self.std) != self.n_outputs:
            raise ValueError(f"Got {len(self.std)} StdSpecs for {self.n_outputs} outputs.")

    @property
    def n_inputs(self):
        return self.layers[0].n_in

    @property
    def n_outputs(self):
        for layer in reversed(self.layers):
            if not isinstance(layer, ReLU):
                return layer.n_out

    def forward(self, X, return_all=False):
        """Evaluate the network with numpy.

        Args:
            X: Inputs, shape (n_samples, n_inputs).
            return_all: If True, also return the output of every layer.

        Returns:
            Outputs, shape (n_samples, n_outputs), and the list of layer outputs
            if ``return_all``.
        """
        x = np.atleast_2d(np.asarray(X, dtype=float))
        z, zs = x, []
        for layer in self.layers:
            if isinstance(layer, Affine):
                z = z @ layer.W + layer.b + (0.0 if layer.Wx is None else x @ layer.Wx)
            elif isinstance(layer, (PiecewiseLinear, Elementwise1D)):
                z = layer(z)
            else:
                s = layer.sign(z.shape[1])
                z = s * np.maximum(s * z, 0.0)
            zs.append(z)
        return (z, zs) if return_all else z

    def encoded(self, input_bounds):
        """The network exactly as the MIP represents it for these input bounds.

        ``Elementwise1D`` layers are replaced by their interpolating
        ``PiecewiseLinear`` layers; every other layer is unchanged. Compare
        ``encoded(bounds).forward(X)`` with ``forward(X)`` to see the approximation.
        """
        if not any(isinstance(layer, Elementwise1D) for layer in self.layers):
            return self
        net = copy.copy(self)
        net.layers = []
        lo, hi = np.asarray(input_bounds, dtype=float).T
        xlo, xhi = lo, hi
        for layer in self.layers:
            if isinstance(layer, Elementwise1D):
                layer = layer.to_piecewise(lo, hi)
            net.layers.append(layer)
            (lo, hi) = propagate_bounds([layer], np.column_stack([lo, hi]), (xlo, xhi))[0]
        return net

    def max_feature_error(self, input_bounds):
        """Bounds on the interpolation error of each ``Elementwise1D`` feature.

        Returns a list with one array per ``Elementwise1D`` layer (empty when the
        network is exact).
        """
        errors = []
        lo, hi = np.asarray(input_bounds, dtype=float).T
        xlo, xhi = lo, hi
        for layer in self.layers:
            if isinstance(layer, Elementwise1D):
                errors.append(layer.max_error(lo, hi))
                layer = layer.to_piecewise(lo, hi)
            (lo, hi) = propagate_bounds([layer], np.column_stack([lo, hi]), (xlo, xhi))[0]
        return errors

    def predict_std(self, X):
        """Evaluate the std of each output with numpy, shape (n_samples, n_outputs)."""
        if self.std is None:
            raise ValueError("This network has no uncertainty (std is None).")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        _, zs = self.forward(X, return_all=True)
        return np.column_stack([s(X if s.tap == -1 else zs[s.tap]) for s in self.std])

    def scale_inputs(self, mean, scale):
        """Network on raw inputs ``x`` for one trained on ``(x - mean) / scale``."""
        mean = np.asarray(mean, dtype=float).reshape(-1)
        scale = np.asarray(scale, dtype=float).reshape(-1)
        if np.any(scale <= 0):
            raise ValueError("Input scales must be positive.")
        net = copy.deepcopy(self)
        shift = mean / scale

        first = net.layers[0]
        if isinstance(first, Elementwise1D):
            raise ValueError("scale_inputs does not support an Elementwise1D first layer.")
        first.b = first.b - shift @ first.W
        first.W = first.W / scale[:, None]
        if isinstance(first, PiecewiseLinear):
            first.breakpoints = [bp * s + m for bp, s, m in zip(first.breakpoints, scale, mean)]
        for layer in net.layers:
            if isinstance(layer, Affine) and layer.Wx is not None:
                layer.b = layer.b - shift @ layer.Wx
                layer.Wx = layer.Wx / scale[:, None]
        for spec in net.std or []:
            if spec.tap == -1:
                spec.c = spec.c - spec.A @ shift
                spec.A = spec.A / scale[None, :]
        return net

    def scale_outputs(self, mean, scale):
        """Network predicting ``y * scale + mean`` (and std times ``|scale|``).

        The scaling is folded into the last layer, unless that layer is a ReLU or
        Elementwise1D, or a ``StdSpec`` taps it; then a new ``Affine`` layer is
        appended so the tapped features are unchanged.
        """
        net = copy.deepcopy(self)
        mean = np.broadcast_to(np.asarray(mean, dtype=float).reshape(-1), (net.n_outputs,))
        scale = np.broadcast_to(np.asarray(scale, dtype=float).reshape(-1), (net.n_outputs,))
        last = net.layers[-1]
        taps_last = any(spec.tap == len(net.layers) - 1 for spec in net.std or [])
        if isinstance(last, (ReLU, Elementwise1D)) or taps_last:
            net.layers.append(Affine(np.diag(scale), mean))
        else:
            last.W = last.W * scale
            last.b = last.b * scale + mean
            if isinstance(last, PiecewiseLinear):
                last.values = [v * scale for v in last.values]
            elif last.Wx is not None:
                last.Wx = last.Wx * scale
        for spec, s in zip(net.std or [], np.abs(scale)):
            spec.A, spec.c, spec.d = spec.A * s, spec.c * s, spec.d * s**2
        return net


@dataclass
class Leaf:
    """A box ``lower < x <= upper`` on which the prediction is ``x @ w + c``.

    Bounds may be infinite. This matches sklearn trees, which send ``x <= threshold``
    to the left child.
    """

    lower: np.ndarray
    upper: np.ndarray
    w: np.ndarray
    c: float

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float).reshape(-1)
        self.upper = np.asarray(self.upper, dtype=float).reshape(-1)
        self.w = np.asarray(self.w, dtype=float).reshape(-1)
        self.c = float(self.c)


@dataclass
class TreeModel:
    """A piecewise-affine model on a partition of the input space into boxes.

    Args:
        leaves: The ``Leaf`` boxes.
        split_gap: In the MIP, a leaf's strict lower bounds ``x > lower`` become
            ``x >= lower + split_gap``. With the default 0 a point exactly on a split
            may take either neighboring leaf's value (the tree takes the left one);
            a positive gap rules that out, at the cost of excluding inputs within the
            gap above each split. It must be well above the solver's feasibility and
            integrality tolerances (about 1e-6), e.g. 1e-4.
    """

    leaves: list
    split_gap: float = 0.0

    def __post_init__(self):
        if not self.leaves:
            raise ValueError("A TreeModel needs at least one leaf.")
        n = self.n_inputs
        for leaf in self.leaves:
            if not (len(leaf.lower) == len(leaf.upper) == len(leaf.w) == n):
                raise ValueError(f"Every leaf needs bounds and weights for {n} inputs.")

    @property
    def n_inputs(self):
        return len(self.leaves[0].w)

    n_outputs = 1
    std = None

    def forward(self, X):
        """Evaluate with numpy, shape (n_samples, 1)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.full(len(X), np.nan)
        for leaf in self.leaves:
            inside = np.all((X > leaf.lower) & (X <= leaf.upper), axis=1)
            out[inside] = X[inside] @ leaf.w + leaf.c
        return out[:, None]


NUMPY_OPS = {
    "exp": np.exp,
    "ln": np.log,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "pow": np.power,
}


def pyomo_ops():
    """The functions of ``NUMPY_OPS`` as Pyomo expression builders."""
    import pyomo.environ as pyo

    return {
        "exp": pyo.exp,
        "ln": pyo.log,
        "log": pyo.log,
        "sqrt": pyo.sqrt,
        "abs": abs,
        "sin": pyo.sin,
        "cos": pyo.cos,
        "tan": pyo.tan,
        "pow": lambda a, b: a**b,
    }


def eval_expression(expression, names, values, ops):
    """Evaluate a formula string such as ``"2.1*(x0*x1) - exp(x1)"``.

    Args:
        expression: The formula, using ``names`` and the functions in ``ops``.
        names: Variable names, e.g. ``["x0", "x1"]``.
        values: One value per name: numpy arrays or Pyomo variables/expressions.
        ops: ``NUMPY_OPS`` or ``pyomo_ops()``.
    """
    context = dict(ops)
    context.update(zip(names, values))
    return eval(expression, {"__builtins__": {}}, context)


@dataclass
class SymbolicModel:
    """A model given by explicit formulas, exported as exact nonlinear expressions.

    This is not a MILP: the Pyomo block holds ``y[j] == f_j(x)`` with the original
    nonlinear functions (division, exp, log, ...), so it needs a nonlinear solver,
    e.g. Gurobi (``pyo.SolverFactory("gurobi_direct_minlp")``), BARON, SCIP or Ipopt
    (local only).

    Args:
        n_inputs: Number of inputs.
        outputs: ``outputs(x, ops)`` returns a list with one expression per output,
            built from the list of inputs ``x`` and the functions in ``ops``.
        variance: Optional ``variance(x, ops)`` returning one predictive variance
            expression per output.
    """

    n_inputs: int
    outputs: object
    variance: object = None

    @property
    def std(self):
        return self.variance

    def forward(self, X):
        """Evaluate with numpy, shape (n_samples, n_outputs)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        cols = [X[:, i] for i in range(self.n_inputs)]
        out = [
            np.broadcast_to(np.asarray(v, dtype=float), (len(X),))
            for v in self.outputs(cols, NUMPY_OPS)
        ]
        return np.column_stack(out)

    def predict_std(self, X):
        """Evaluate the std of each output with numpy, shape (n_samples, n_outputs)."""
        if self.variance is None:
            raise ValueError("This model has no uncertainty (variance is None).")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        cols = [X[:, i] for i in range(self.n_inputs)]
        var = [
            np.broadcast_to(np.asarray(v, dtype=float), (len(X),))
            for v in self.variance(cols, NUMPY_OPS)
        ]
        return np.sqrt(np.maximum(np.column_stack(var), 0.0))


def _add_symbolic(block, model, x, include_std):
    """``y[j] == f_j(x)`` (and the variance) as nonlinear Pyomo expressions."""
    import pyomo.environ as pyo

    ops = pyomo_ops()
    exprs = list(model.outputs(x, ops))
    block.y = pyo.Var(range(len(exprs)))
    block.y_def = pyo.Constraint(range(len(exprs)), rule=lambda b, j: b.y[j] == exprs[j])
    if model.variance is None:
        return
    variance = list(model.variance(x, ops))
    block.variance = pyo.Expression(range(len(variance)), rule=lambda b, j: variance[j])
    if include_std:
        block.std = pyo.Var(range(len(variance)), within=pyo.NonNegativeReals)
        if include_std == "exact":
            rule = lambda b, j: b.std[j] ** 2 == b.variance[j]  # noqa: E731
        else:
            rule = lambda b, j: b.std[j] ** 2 >= b.variance[j]  # noqa: E731
        block.std_def = pyo.Constraint(range(len(variance)), rule=rule)


def _affine_interval(W, lo, hi):
    """Bounds of ``z @ W`` for ``lo <= z <= hi``."""
    Wp, Wn = np.maximum(W, 0.0), np.minimum(W, 0.0)
    return lo @ Wp + hi @ Wn, hi @ Wp + lo @ Wn


def _pwl_bounds(layer, lo, hi):
    """Bounds of a PiecewiseLinear layer for ``lo <= z <= hi``."""
    out_lo, out_hi = _affine_interval(layer.W, lo, hi)
    out_lo, out_hi = out_lo + layer.b, out_hi + layer.b
    for i in range(layer.n_in):
        _, vals = layer.segments(i, lo[i], hi[i])
        out_lo = out_lo + vals.min(axis=0)
        out_hi = out_hi + vals.max(axis=0)
    return out_lo, out_hi


def propagate_bounds(layers, input_bounds, network_input_bounds=None):
    """Interval bounds on the output of every layer.

    Args:
        layers: List of ``Affine`` / ``ReLU`` / ``PiecewiseLinear`` layers
            (``Elementwise1D`` layers are bounded through their interpolation).
        input_bounds: Array-like of (lower, upper) per input, shape (n_inputs, 2).
        network_input_bounds: (lower, upper) arrays of the network inputs, for skip
            connections when ``layers`` is not the whole network.

    Returns:
        List of (lower, upper) array pairs, one per layer.
    """
    bounds = np.asarray(input_bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    if np.any(lo > hi):
        raise ValueError("Each input lower bound must not exceed its upper bound.")
    xlo, xhi = (lo, hi) if network_input_bounds is None else network_input_bounds
    out = []
    for layer in layers:
        if isinstance(layer, Affine):
            new_lo, new_hi = _affine_interval(layer.W, lo, hi)
            if layer.Wx is not None:
                skip_lo, skip_hi = _affine_interval(layer.Wx, xlo, xhi)
                new_lo, new_hi = new_lo + skip_lo, new_hi + skip_hi
            lo, hi = new_lo + layer.b, new_hi + layer.b
        elif isinstance(layer, PiecewiseLinear):
            lo, hi = _pwl_bounds(layer, lo, hi)
        elif isinstance(layer, Elementwise1D):
            lo, hi = _pwl_bounds(layer.to_piecewise(lo, hi), lo, hi)
        else:
            s = layer.sign(len(lo))
            relu_lo, relu_hi = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
            neg_lo, neg_hi = np.minimum(lo, 0.0), np.minimum(hi, 0.0)
            lo, hi = np.where(s > 0, relu_lo, neg_lo), np.where(s > 0, relu_hi, neg_hi)
        out.append((lo, hi))
    return out


def check_convex(network, tol=1e-12):
    """Raise ``ValueError`` unless ``formulation="convex"`` is exact for ``network``.

    That needs plain ReLUs (no concave units, no splines) and nonnegative weights on
    every hidden unit after the first ReLU, which makes the output convex and
    nondecreasing in every ReLU output.
    """
    seen_relu = False
    for k, layer in enumerate(network.layers):
        if isinstance(layer, (PiecewiseLinear, Elementwise1D)):
            raise ValueError(
                f"formulation='convex' does not support {type(layer).__name__} layers."
            )
        if isinstance(layer, ReLU):
            if layer.concave is not None and layer.concave.any():
                raise ValueError("formulation='convex' needs convex ReLUs (no concave units).")
            seen_relu = True
        elif seen_relu and np.any(layer.W < -tol):
            raise ValueError(
                f"formulation='convex' needs nonnegative weights after the first ReLU, but "
                f"layer {k} has a weight of {layer.W.min():.3g}. The network is not "
                "input-convex; use formulation='bigm'."
            )


def _resolve_input_bounds(n_inputs, input_bounds, inputs):
    """Input bounds from the argument, or from the bounds of the given Pyomo vars."""
    if input_bounds is None:
        if inputs is None:
            raise ValueError(
                "input_bounds is required: the MIP encoding needs finite bounds on every "
                "input. Pass input_bounds=[(lo, hi), ...] or bounded Pyomo vars as inputs."
            )
        input_bounds = [v.bounds for v in inputs]
    bounds = np.array(input_bounds, dtype=float)
    if bounds.shape != (n_inputs, 2):
        raise ValueError(f"input_bounds must have shape ({n_inputs}, 2), got {bounds.shape}.")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Every input needs finite lower and upper bounds.")
    return bounds


def add_to_pyomo(
    model, block=None, inputs=None, input_bounds=None, include_std=False, formulation="bigm"
):
    """Write a ``MIPNetwork``, ``TreeModel`` or ``SymbolicModel`` into a Pyomo block.

    Networks and trees become MILP constraints; a ``SymbolicModel`` becomes exact
    nonlinear constraints (see its docstring).

    Args:
        model: The ``MIPNetwork``, ``TreeModel`` or ``SymbolicModel`` to encode.
        block: Pyomo block (or ConcreteModel) to populate. A new ConcreteModel is
            created if None.
        inputs: Optional list of existing Pyomo vars to use as the inputs, e.g. the
            decision variables of your own model. If None, ``block.x`` is created.
        input_bounds: (lower, upper) per input. If None, the bounds of ``inputs`` are
            used. Required either way: they set the big-M constants.
        include_std: Add ``block.std[j] >= 0`` for models with uncertainty
            (``block.variance`` is always added). ``True`` or ``"cone"`` adds the
            second-order cone ``block.std[j] >= sqrt(block.variance[j])``: convex, and
            exact when the objective penalizes the std (minimize ``y + k * std`` or
            maximize ``y - k * std``). ``"exact"`` adds ``block.std[j]**2 ==
            block.variance[j]``, a nonconvex quadratic needed when the objective
            rewards the std (e.g. maximize ``y + k * std`` for exploration). Both need
            a solver for quadratic constraints, e.g. Gurobi.
        formulation: ``"bigm"`` (exact MILP) or ``"convex"`` (LP without binaries;
            exact for input-convex networks whose output is minimized or bounded
            above, see ``check_convex``). The variance is not exported with
            ``"convex"``, since the hidden units are then only bounded below.

    Returns:
        The populated block, with ``block.x`` (the inputs) and ``block.y`` (the outputs)
        indexed by position.
    """
    import pyomo.environ as pyo

    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}, got {formulation!r}.")
    if include_std not in (False, True, "cone", "exact"):
        raise ValueError(
            f"include_std must be False, True, 'cone' or 'exact', got {include_std!r}."
        )
    if block is None:
        block = pyo.ConcreteModel()

    bounds = _resolve_input_bounds(model.n_inputs, input_bounds, inputs)
    if inputs is None:
        block.x = pyo.Var(range(model.n_inputs), bounds=lambda b, i: tuple(bounds[i]))
        x = [block.x[i] for i in range(model.n_inputs)]
    else:
        if len(inputs) != model.n_inputs:
            raise ValueError(f"Got {len(inputs)} inputs, the model needs {model.n_inputs}.")
        x = list(inputs)
        block.x = pyo.Reference(x)

    if isinstance(model, TreeModel):
        if formulation != "bigm":
            raise ValueError("Trees support only formulation='bigm'.")
        _add_tree(block, model, x, bounds)
        return block
    if isinstance(model, SymbolicModel):
        if formulation != "bigm":
            raise ValueError("Symbolic models support only the default formulation.")
        _add_symbolic(block, model, x, include_std)
        return block

    if formulation == "convex":
        check_convex(model)
        if include_std:
            raise ValueError("include_std is not available with formulation='convex'.")

    model = model.encoded(bounds)  # interpolate any Elementwise1D layers on these bounds
    layer_bounds = propagate_bounds(model.layers, bounds)
    taps = {-1: x}
    z, z_bounds = x, (bounds[:, 0], bounds[:, 1])
    for k, layer in enumerate(model.layers):
        lo, hi = layer_bounds[k]
        idx = range(len(lo))
        name = "y" if k == len(model.layers) - 1 else f"z{k}"
        var = pyo.Var(idx, bounds=lambda b, i, lo=lo, hi=hi: (lo[i], hi[i]))
        block.add_component(name, var)

        if isinstance(layer, Affine):
            _add_affine(block, name, var, z, x, layer)
        elif isinstance(layer, PiecewiseLinear):
            _add_pwl(block, name, var, z, z_bounds, layer)
        else:
            _add_relu(block, name, var, z, z_bounds, layer.sign(len(z)), formulation)

        z, z_bounds = [var[i] for i in idx], (lo, hi)
        taps[k] = z

    if model.std is not None and formulation == "bigm":
        _add_std(block, model.std, taps, include_std)

    return block


def _linear(W, z, j):
    """``sum_i W[i, j] * z[i]``, skipping zero weights."""
    return sum(W[i, j] * z[i] for i in range(len(z)) if W[i, j] != 0)


def _add_affine(block, name, var, z, x, layer):
    """``var = z @ W + x @ Wx + b``."""
    import pyomo.environ as pyo

    def rule(b, j):
        skip = 0 if layer.Wx is None else _linear(layer.Wx, x, j)
        return var[j] == _linear(layer.W, z, j) + skip + layer.b[j]

    block.add_component(f"{name}_def", pyo.Constraint(range(layer.n_out), rule=rule))


def _add_relu(block, name, var, z, pre_bounds, sign, formulation):
    """Big-M encoding of ``var = s * max(0, s * z)`` given the bounds on ``z``.

    With ``s = -1`` (a concave unit) this is ``var = min(0, z)``; the encoding is the
    ReLU one applied to ``s * var`` and ``s * z``, whose bounds are ``s``-flipped.
    """
    import pyomo.environ as pyo

    lo, hi = pre_bounds
    idx = list(range(len(z)))
    flo = np.where(sign > 0, lo, -hi)  # bounds of s * z
    fhi = np.where(sign > 0, hi, -lo)
    mixed = [i for i in idx if flo[i] < 0 < fhi[i]] if formulation == "bigm" else []
    on = pyo.Var(mixed, within=pyo.Binary)
    block.add_component(f"{name}_on", on)

    cons = pyo.ConstraintList()
    block.add_component(f"{name}_relu", cons)
    for i in idx:
        s = float(sign[i])
        out, pre = s * var[i], s * z[i]
        if fhi[i] <= 0:
            cons.add(var[i] == 0)
        elif flo[i] >= 0:
            cons.add(var[i] == z[i])
        elif formulation == "convex":
            cons.add(out >= pre)  # out >= 0 comes from the var bounds
        else:
            cons.add(out >= pre)
            cons.add(out <= pre - flo[i] * (1 - on[i]))
            cons.add(out <= fhi[i] * on[i])
    # s * var >= 0 comes from the var bounds (from propagate_bounds).


def _add_pwl(block, name, var, z, pre_bounds, layer):
    """``var[j] = sum_i f_ij(z[i]) + z @ W + b`` with one SOS2 (lambda) set per input.

    The breakpoints are restricted to the bounds of each input, so the constant
    pieces outside the original breakpoints (clipping) are represented exactly.
    """
    import pyomo.environ as pyo

    lo, hi = pre_bounds
    pieces = {}
    for i in range(layer.n_in):
        pts, vals = layer.segments(i, lo[i], hi[i])
        K = len(pts)
        if K == 1 or not np.any(vals):  # a fixed input, or no nonlinear part
            pieces[i] = vals[0]
            continue
        lam = pyo.Var(range(K), bounds=(0, 1))
        block.add_component(f"{name}_lam{i}", lam)
        cons = pyo.ConstraintList()
        block.add_component(f"{name}_sos{i}", cons)
        cons.add(sum(lam[k] for k in range(K)) == 1)
        cons.add(z[i] == sum(float(pts[k]) * lam[k] for k in range(K)))
        if K > 2:
            seg = pyo.Var(range(K - 1), within=pyo.Binary)
            block.add_component(f"{name}_seg{i}", seg)
            cons.add(sum(seg[s] for s in range(K - 1)) == 1)
            for k in range(K):
                adjacent = [seg[s] for s in (k - 1, k) if 0 <= s < K - 1]
                cons.add(lam[k] <= sum(adjacent))
        pieces[i] = [sum(float(vals[k, j]) * lam[k] for k in range(K)) for j in range(layer.n_out)]

    def rule(b, j):
        return var[j] == sum(pieces[i][j] for i in range(layer.n_in)) + _linear(
            layer.W, z, j
        ) + float(layer.b[j])

    block.add_component(f"{name}_def", pyo.Constraint(range(layer.n_out), rule=rule))


def _add_std(block, std_specs, taps, include_std):
    """Add the variance expressions (and optionally the SOC std vars)."""
    import pyomo.environ as pyo

    n_out = len(std_specs)
    variance = {}
    for j, spec in enumerate(std_specs):
        z = taps[spec.tap]
        k_idx = range(spec.A.shape[0])
        r = pyo.Var(k_idx)
        block.add_component(f"std_r{j}", r)

        def r_rule(b, i, z=z, r=r, A=spec.A, c=spec.c):
            return r[i] == sum(A[i, m] * z[m] for m in range(len(z)) if A[i, m] != 0) + c[i]

        block.add_component(f"std_r{j}_def", pyo.Constraint(k_idx, rule=r_rule))
        variance[j] = sum(r[i] ** 2 for i in k_idx) + spec.d

    block.variance = pyo.Expression(range(n_out), rule=lambda b, j: variance[j])
    if include_std:
        # Standard cone form ||(r, sqrt(d))|| <= std: the constant goes in through its own
        # variable so solvers recognize a second-order cone, not a general quadratic.
        block.std = pyo.Var(range(n_out), within=pyo.NonNegativeReals)
        block.std_const = pyo.Var(range(n_out))
        block.std_const_def = pyo.Constraint(
            range(n_out), rule=lambda b, j: b.std_const[j] == float(np.sqrt(std_specs[j].d))
        )

        def soc_rule(b, j):
            r = b.component(f"std_r{j}")
            norm2 = sum(r[i] ** 2 for i in r) + b.std_const[j] ** 2
            if include_std == "exact":
                return norm2 == b.std[j] ** 2
            return norm2 <= b.std[j] ** 2

        block.std_soc = pyo.Constraint(range(n_out), rule=soc_rule)


def _add_tree(block, tree, x, bounds):
    """One binary per reachable leaf; big-M constants from each leaf's box."""
    import pyomo.environ as pyo

    L, U = bounds[:, 0], bounds[:, 1]
    leaves = []
    if tree.split_gap < 0:
        raise ValueError(f"split_gap must be nonnegative, got {tree.split_gap}.")
    for leaf in tree.leaves:
        a, b = np.maximum(leaf.lower + tree.split_gap, L), np.minimum(leaf.upper, U)
        if np.all(a <= b):
            leaves.append((leaf, a, b))
    if not leaves:
        raise ValueError("No leaf of the tree intersects the input bounds.")

    n = len(leaves)
    block.leaf = pyo.Var(range(n), within=pyo.Binary)
    block.leaf_one = pyo.Constraint(expr=sum(block.leaf[k] for k in range(n)) == 1)

    g_box, g_all = [], []
    for leaf, a, b in leaves:
        lo_box, hi_box = _affine_interval(leaf.w[:, None], a, b)
        lo_all, hi_all = _affine_interval(leaf.w[:, None], L, U)
        g_box.append((lo_box[0] + leaf.c, hi_box[0] + leaf.c))
        g_all.append((lo_all[0] + leaf.c, hi_all[0] + leaf.c))

    block.y_leaf = pyo.Var(
        range(n), bounds=lambda m, k: (min(g_box[k][0], 0.0), max(g_box[k][1], 0.0))
    )
    y_lo = min(g[0] for g in g_box)
    y_hi = max(g[1] for g in g_box)
    block.y = pyo.Var(range(1), bounds=(y_lo, y_hi))
    block.y_def = pyo.Constraint(expr=block.y[0] == sum(block.y_leaf[k] for k in range(n)))

    cons = pyo.ConstraintList()
    block.tree_cons = cons
    for k, (leaf, a, b) in enumerate(leaves):
        on = block.leaf[k]
        for f in range(len(x)):
            if b[f] < U[f]:
                cons.add(x[f] <= b[f] + (U[f] - b[f]) * (1 - on))
            if a[f] > L[f]:
                cons.add(x[f] >= a[f] - (a[f] - L[f]) * (1 - on))
        g = sum(float(leaf.w[f]) * x[f] for f in range(len(x)) if leaf.w[f] != 0) + leaf.c
        yl = block.y_leaf[k]
        cons.add(yl <= g_box[k][1] * on)
        cons.add(yl >= g_box[k][0] * on)
        cons.add(yl - g <= max(0.0, -g_all[k][0]) * (1 - on))
        cons.add(yl - g >= -max(0.0, g_all[k][1]) * (1 - on))


class MIPExportMixin:
    """Adds ``to_pyomo()`` to an estimator that implements ``mip_network()``.

    ``mip_network()`` must return the trained model as a ``MIPNetwork`` (or
    ``TreeModel``) in the original (unscaled) input and output units, and raise
    ``ValueError`` when the configuration is not piecewise-linear (e.g. a smooth
    activation). An estimator can also define ``_mip_default_input_bounds()`` to
    supply bounds when the caller gives none.
    """

    def mip_network(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _mip_default_input_bounds(self):
        return None

    def to_pyomo(
        self,
        input_bounds=None,
        block=None,
        inputs=None,
        include_std=False,
        formulation="bigm",
        **options,
    ):
        """Export the trained model as Pyomo MILP constraints.

        Args:
            input_bounds: (lower, upper) per input feature. Required unless ``inputs``
                are bounded Pyomo vars. Tighter bounds give faster solves.
            block: Pyomo block to populate (e.g. ``m.surrogate = pyo.Block()``).
                A new ConcreteModel is created and returned if None.
            inputs: Existing Pyomo vars to use as the model inputs.
            include_std: Also add ``block.std`` (models with uncertainty only):
                ``True``/``"cone"`` for a second-order cone, exact when the objective
                penalizes the std; ``"exact"`` for a nonconvex equality, needed when
                it rewards the std. See ``pycse.sklearn.mip.add_to_pyomo``.
            formulation: ``"bigm"`` (exact MILP) or ``"convex"`` (an LP for
                input-convex networks whose output is minimized or bounded above).
            **options: Model-specific options passed to ``mip_network()``, e.g.
                ``split_gap`` for ``LeafModelRegressor`` or ``points_per_period``
                for ``JAXPeriodicRegressor``.

        Returns:
            The populated block. ``block.x[i]`` are the inputs, ``block.y[j]`` the
            predictions, and ``block.variance[j]`` the predictive variance when the
            model has uncertainty.
        """
        network = self.mip_network(**options)
        if include_std and network.std is None:
            raise ValueError(f"{type(self).__name__} has no uncertainty to export.")
        if input_bounds is None and inputs is None:
            input_bounds = self._mip_default_input_bounds()
        return add_to_pyomo(
            network,
            block=block,
            inputs=inputs,
            input_bounds=input_bounds,
            include_std=include_std,
            formulation=formulation,
        )


def std_from_covariance(cov, tap, scale=1.0, offset=None, d=0.0):
    """``StdSpec`` for ``std = scale * sqrt((z - offset) @ cov @ (z - offset) + d)``.

    Factors the (symmetric positive semidefinite) ``cov`` with an eigendecomposition
    so that ``(z - offset) @ cov @ (z - offset) = ||A (z - offset)||^2``.
    """
    cov = np.asarray(cov, dtype=float)
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    A = scale * (np.sqrt(np.maximum(w, 0.0))[:, None] * V.T)
    c = None if offset is None else -A @ np.asarray(offset, dtype=float)
    return StdSpec(tap=tap, A=A, c=c, d=scale**2 * d)


def llpr_std(cov, alpha_squared, zeta_squared, tap, y_scale=1.0):
    """``StdSpec``s for LLPR: ``std_j = y_scale_j sqrt(alpha_j f^T (C + zeta_j I)^-1 f)``.

    Args:
        cov: Feature covariance ``F^T F``, shape (n_feat, n_feat).
        alpha_squared, zeta_squared: Scalars or one value per output.
        tap: Layer index of the features ``f``.
        y_scale: Output scale(s) when the targets were standardized.

    Returns:
        A list of ``StdSpec``, one per output.
    """
    cov = np.asarray(cov, dtype=float)
    alpha2 = np.atleast_1d(np.asarray(alpha_squared, dtype=float))
    zeta2 = np.atleast_1d(np.asarray(zeta_squared, dtype=float))
    n_out = max(len(alpha2), len(zeta2), np.size(y_scale))
    alpha2, zeta2 = np.broadcast_to(alpha2, n_out), np.broadcast_to(zeta2, n_out)
    y_scale = np.broadcast_to(np.abs(np.asarray(y_scale, dtype=float)).reshape(-1), n_out)
    eye = np.eye(cov.shape[0])
    return [
        std_from_covariance(np.linalg.inv(cov + z2 * eye), tap, scale=s * np.sqrt(a2))
        for a2, z2, s in zip(alpha2, zeta2, y_scale)
    ]


def affine_from_predict(predict, x0, step, n_check=20, rtol=1e-6, seed=0):
    """Recover ``f(x) = x @ w + c`` from a model that is affine, by probing ``predict``.

    The slope along each input comes from a finite difference of size ``step``; the
    result is then checked at ``n_check`` random points so that a nonlinear model
    (e.g. a pipeline with ``PolynomialFeatures``) raises instead of being linearized.

    Args:
        predict: Function mapping an (n, n_inputs) array to n predictions.
        x0: Center point for the probes, shape (n_inputs,).
        step: Probe step per input, shape (n_inputs,) (positive).

    Returns:
        ``(w, c)``.

    Raises:
        ValueError: If ``predict`` is not affine.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    step = np.asarray(step, dtype=float).reshape(-1)
    n = len(x0)

    def f(X):
        return np.asarray(predict(X), dtype=float).reshape(-1)

    probes = np.vstack([x0, x0 + np.diag(step)])
    fp = f(probes)
    w = (fp[1:] - fp[0]) / step
    c = fp[0] - x0 @ w

    X = x0 + np.random.default_rng(seed).uniform(-3, 3, size=(n_check, n)) * step
    expected = X @ w + c
    scale = np.max(np.abs(expected)) + np.max(np.abs(fp)) + 1.0
    if np.max(np.abs(f(X) - expected)) > rtol * scale:
        raise ValueError("The model is not affine in its inputs, so it is not MIP-representable.")
    return w, c
