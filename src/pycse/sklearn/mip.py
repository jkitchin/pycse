"""Mixed-integer programming (MIP) export of trained piecewise-linear models.

A trained network built from affine maps and ReLUs is an exactly piecewise-linear
function, so it can be embedded in an optimization problem as mixed-integer linear
constraints. This module holds the solver-side machinery once, and each estimator
only describes its trained weights:

- ``Affine`` and ``ReLU`` are small layer specs holding numpy arrays.
- ``MIPNetwork`` is a list of layers plus optional uncertainty (``StdSpec``).
- ``propagate_bounds`` computes interval bounds for every layer, which give the
  big-M constants of the ReLU encoding.
- ``add_to_pyomo`` writes a ``MIPNetwork`` into a Pyomo block.
- ``MIPExportMixin`` gives an estimator ``to_pyomo()`` from its ``mip_network()``.

The ReLU ``y = max(0, a)`` with bounds ``L <= a <= U`` (``L < 0 < U``) is encoded
with one binary ``d``::

    y >= 0,  y >= a,  y <= a - L (1 - d),  y <= U d

Units that are always active (``L >= 0``) or always inactive (``U <= 0``) need no
binary. Tight input bounds give tight big-M constants and faster solves.

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

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Affine:
    """Affine layer ``z_out = z_in @ W + b``.

    Args:
        W: Weight matrix, shape (n_in, n_out).
        b: Bias vector, shape (n_out,).
    """

    W: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        self.W = np.atleast_2d(np.asarray(self.W, dtype=float))
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        if self.W.shape[1] != self.b.shape[0]:
            raise ValueError(f"W has {self.W.shape[1]} outputs but b has {self.b.shape[0]}.")

    @property
    def n_in(self):
        return self.W.shape[0]

    @property
    def n_out(self):
        return self.W.shape[1]


@dataclass
class ReLU:
    """Elementwise ``max(0, z)``."""


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
        layers: List of ``Affine`` and ``ReLU`` layers; the first must be ``Affine``.
        std: Optional list of ``StdSpec``, one per output.
    """

    layers: list
    std: list = field(default=None)

    def __post_init__(self):
        if not self.layers or not isinstance(self.layers[0], Affine):
            raise ValueError("A MIPNetwork must start with an Affine layer.")
        n = self.layers[0].n_out
        for layer in self.layers[1:]:
            if isinstance(layer, Affine):
                if layer.n_in != n:
                    raise ValueError(f"Layer expects {layer.n_in} inputs but receives {n}.")
                n = layer.n_out
            elif not isinstance(layer, ReLU):
                raise TypeError(f"Unsupported layer type: {type(layer).__name__}")
        if self.std is not None and len(self.std) != self.n_outputs:
            raise ValueError(f"Got {len(self.std)} StdSpecs for {self.n_outputs} outputs.")

    @property
    def n_inputs(self):
        return self.layers[0].n_in

    @property
    def n_outputs(self):
        for layer in reversed(self.layers):
            if isinstance(layer, Affine):
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
        z = np.atleast_2d(np.asarray(X, dtype=float))
        zs = []
        for layer in self.layers:
            z = z @ layer.W + layer.b if isinstance(layer, Affine) else np.maximum(z, 0.0)
            zs.append(z)
        return (z, zs) if return_all else z

    def predict_std(self, X):
        """Evaluate the std of each output with numpy, shape (n_samples, n_outputs)."""
        if self.std is None:
            raise ValueError("This network has no uncertainty (std is None).")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        _, zs = self.forward(X, return_all=True)
        return np.column_stack([s(X if s.tap == -1 else zs[s.tap]) for s in self.std])


def propagate_bounds(layers, input_bounds):
    """Interval bounds on the output of every layer.

    Args:
        layers: List of ``Affine`` / ``ReLU`` layers.
        input_bounds: Array-like of (lower, upper) per input, shape (n_inputs, 2).

    Returns:
        List of (lower, upper) array pairs, one per layer.
    """
    bounds = np.asarray(input_bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    if np.any(lo > hi):
        raise ValueError("Each input lower bound must not exceed its upper bound.")
    out = []
    for layer in layers:
        if isinstance(layer, Affine):
            Wp, Wn = np.maximum(layer.W, 0.0), np.minimum(layer.W, 0.0)
            lo, hi = lo @ Wp + hi @ Wn + layer.b, hi @ Wp + lo @ Wn + layer.b
        else:
            lo, hi = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
        out.append((lo, hi))
    return out


def _resolve_input_bounds(n_inputs, input_bounds, inputs):
    """Input bounds from the argument, or from the bounds of the given Pyomo vars."""
    if input_bounds is None:
        if inputs is None:
            raise ValueError(
                "input_bounds is required: the MIP encoding needs finite bounds on every "
                "input. Pass input_bounds=[(lo, hi), ...] or bounded Pyomo vars as inputs."
            )
        input_bounds = [v.bounds for v in inputs]
    bounds = np.asarray(input_bounds, dtype=float)
    if bounds.shape != (n_inputs, 2):
        raise ValueError(f"input_bounds must have shape ({n_inputs}, 2), got {bounds.shape}.")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Every input needs finite lower and upper bounds.")
    return bounds


def add_to_pyomo(network, block=None, inputs=None, input_bounds=None, include_std=False):
    """Write a ``MIPNetwork`` into a Pyomo block as MILP constraints.

    Args:
        network: The ``MIPNetwork`` to encode.
        block: Pyomo block (or ConcreteModel) to populate. A new ConcreteModel is
            created if None.
        inputs: Optional list of existing Pyomo vars to use as the inputs, e.g. the
            decision variables of your own model. If None, ``block.x`` is created.
        input_bounds: (lower, upper) per input. If None, the bounds of ``inputs`` are
            used. Required either way: they set the big-M constants.
        include_std: If True, add ``block.std[j] >= 0`` with the second-order cone
            constraint ``block.std[j]**2 >= block.variance[j]`` (needs a solver that
            handles quadratic constraints, e.g. Gurobi). ``block.variance`` is always
            added when the network has uncertainty.

    Returns:
        The populated block, with ``block.x`` (the inputs) and ``block.y`` (the outputs)
        indexed by position.
    """
    import pyomo.environ as pyo

    if block is None:
        block = pyo.ConcreteModel()

    bounds = _resolve_input_bounds(network.n_inputs, input_bounds, inputs)
    layer_bounds = propagate_bounds(network.layers, bounds)

    if inputs is None:
        block.x = pyo.Var(range(network.n_inputs), bounds=lambda b, i: tuple(bounds[i]))
        z_in = [block.x[i] for i in range(network.n_inputs)]
    else:
        if len(inputs) != network.n_inputs:
            raise ValueError(f"Got {len(inputs)} inputs, the network needs {network.n_inputs}.")
        z_in = list(inputs)
        block.x = pyo.Reference(z_in)

    taps = {-1: z_in}
    z = z_in
    for k, layer in enumerate(network.layers):
        lo, hi = layer_bounds[k]
        idx = range(len(lo))
        name = "y" if k == len(network.layers) - 1 else f"z{k}"
        var = pyo.Var(idx, bounds=lambda b, i, lo=lo, hi=hi: (lo[i], hi[i]))
        block.add_component(name, var)

        if isinstance(layer, Affine):
            W, bias = layer.W, layer.b

            def affine_rule(b, j, z=z, var=var, W=W, bias=bias):
                terms = sum(W[i, j] * z[i] for i in range(len(z)) if W[i, j] != 0)
                return var[j] == terms + bias[j]

            block.add_component(f"{name}_def", pyo.Constraint(idx, rule=affine_rule))
        else:
            _add_relu(block, name, var, z, layer_bounds[k - 1])

        z = [var[i] for i in idx]
        taps[k] = z

    if network.std is not None:
        _add_std(block, network.std, taps, include_std)

    return block


def _add_relu(block, name, var, z, pre_bounds):
    """Big-M encoding of ``var = max(0, z)`` given the bounds on ``z``."""
    import pyomo.environ as pyo

    lo, hi = pre_bounds
    idx = list(range(len(z)))
    mixed = [i for i in idx if lo[i] < 0 < hi[i]]
    block.add_component(f"{name}_on", pyo.Var(mixed, within=pyo.Binary))
    on = block.component(f"{name}_on")

    cons = pyo.ConstraintList()
    block.add_component(f"{name}_relu", cons)
    for i in idx:
        if hi[i] <= 0:
            cons.add(var[i] == 0)
        elif lo[i] >= 0:
            cons.add(var[i] == z[i])
        else:
            cons.add(var[i] >= z[i])
            cons.add(var[i] <= z[i] - lo[i] * (1 - on[i]))
            cons.add(var[i] <= hi[i] * on[i])
    # var >= 0 comes from its bounds (the lower bound of a ReLU output is >= 0).


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
        block.std = pyo.Var(range(n_out), within=pyo.NonNegativeReals)
        block.std_soc = pyo.Constraint(
            range(n_out), rule=lambda b, j: b.std[j] ** 2 >= b.variance[j]
        )


class MIPExportMixin:
    """Adds ``to_pyomo()`` to an estimator that implements ``mip_network()``.

    ``mip_network()`` must return the trained model as a ``MIPNetwork`` in the
    original (unscaled) input and output units, and raise ``ValueError`` when the
    configuration is not piecewise-linear (e.g. a smooth activation).
    """

    def mip_network(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def to_pyomo(self, input_bounds=None, block=None, inputs=None, include_std=False):
        """Export the trained model as Pyomo MILP constraints.

        Args:
            input_bounds: (lower, upper) per input feature. Required unless ``inputs``
                are bounded Pyomo vars. Tighter bounds give faster solves.
            block: Pyomo block to populate (e.g. ``m.surrogate = pyo.Block()``).
                A new ConcreteModel is created and returned if None.
            inputs: Existing Pyomo vars to use as the model inputs.
            include_std: Also add ``block.std`` with a second-order cone constraint
                (models with uncertainty only; needs a solver for quadratic
                constraints).

        Returns:
            The populated block. ``block.x[i]`` are the inputs, ``block.y[j]`` the
            predictions, and ``block.variance[j]`` the predictive variance when the
            model has uncertainty.
        """
        network = self.mip_network()
        if include_std and network.std is None:
            raise ValueError(f"{type(self).__name__} has no uncertainty to export.")
        return add_to_pyomo(
            network, block=block, inputs=inputs, input_bounds=input_bounds, include_std=include_std
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
