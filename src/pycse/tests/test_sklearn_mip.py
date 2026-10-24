"""Tests for MIP export of piecewise-linear models (pycse.sklearn.mip).

The core check: fix the inputs of the exported MILP, solve it, and compare the
outputs (and the variance expression) with the model's own predict().
"""

import numpy as np
import pytest

from pycse.sklearn.mip import (
    Affine,
    MIPNetwork,
    ReLU,
    StdSpec,
    add_to_pyomo,
    propagate_bounds,
    std_from_covariance,
)

try:
    import pyomo.environ as pyo

    HIGHS_AVAILABLE = pyo.SolverFactory("highs").available(exception_flag=False)
except ImportError:
    HIGHS_AVAILABLE = False

needs_highs = pytest.mark.skipif(not HIGHS_AVAILABLE, reason="pyomo/highspy not installed")


def random_network(n_in=2, hidden=(6, 5), n_out=1, seed=0, relu_last=False):
    rng = np.random.default_rng(seed)
    layers, n = [], n_in
    for h in hidden:
        layers += [Affine(rng.normal(size=(n, h)), rng.normal(size=h)), ReLU()]
        n = h
    layers.append(Affine(rng.normal(size=(n, n_out)), rng.normal(size=n_out)))
    if relu_last:
        layers.append(ReLU())
    return MIPNetwork(layers)


def solve_at(block, x):
    """Fix the block inputs to x and solve; return outputs and variances."""
    for i, xi in enumerate(np.ravel(x)):
        block.x[i].fix(float(xi))
    if not hasattr(block, "_test_obj"):
        block._test_obj = pyo.Objective(expr=0.0)
    result = pyo.SolverFactory("highs").solve(block)
    assert str(result.solver.termination_condition) == "optimal"
    y = np.array([pyo.value(block.y[j]) for j in block.y])
    var = (
        np.array([pyo.value(block.variance[j]) for j in block.variance])
        if hasattr(block, "variance")
        else None
    )
    return y, var


def check_fixed_inputs(model_or_block, network, X, bounds, predict, predict_std=None):
    """Solve the MILP at each row of X and compare with predict / predict_std."""
    if hasattr(model_or_block, "to_pyomo"):
        block = model_or_block.to_pyomo(input_bounds=bounds)
    else:
        block = add_to_pyomo(network, input_bounds=bounds)
    for x in X:
        y, var = solve_at(block, x)
        np.testing.assert_allclose(y, np.ravel(predict(x[None, :])), rtol=1e-5, atol=1e-5)
        if predict_std is not None:
            np.testing.assert_allclose(
                np.sqrt(var), np.ravel(predict_std(x[None, :])), rtol=1e-4, atol=1e-6
            )


# ---------------------------------------------------------------------------
# Generic module
# ---------------------------------------------------------------------------


class TestNetwork:
    def test_validation(self):
        with pytest.raises(ValueError, match="start with an Affine"):
            MIPNetwork([ReLU()])
        with pytest.raises(ValueError, match="expects 3 inputs"):
            MIPNetwork([Affine(np.ones((2, 4)), np.zeros(4)), Affine(np.ones((3, 1)), [0])])
        with pytest.raises(ValueError, match="outputs but b"):
            Affine(np.ones((2, 3)), np.zeros(2))

    def test_forward(self):
        net = MIPNetwork([Affine([[1.0, -1.0]], [0.0, 0.5]), ReLU(), Affine([[2.0], [3.0]], [1])])
        np.testing.assert_allclose(net.forward([[1.0], [-1.0]]), [[3.0], [5.5]])

    def test_bounds_are_sound(self):
        net = random_network(n_in=3, hidden=(8, 8), seed=1)
        bounds = np.array([[-1, 1], [0, 2], [-3, -1]], dtype=float)
        lb = propagate_bounds(net.layers, bounds)
        rng = np.random.default_rng(2)
        X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(2000, 3))
        _, zs = net.forward(X, return_all=True)
        for z, (lo, hi) in zip(zs, lb):
            assert np.all(z >= lo - 1e-9) and np.all(z <= hi + 1e-9)


@needs_highs
class TestPyomo:
    def test_fixed_inputs_match_forward(self):
        net = random_network(n_in=2, hidden=(6, 5), n_out=2, seed=3)
        bounds = [(-1, 1), (-2, 2)]
        X = np.random.default_rng(4).uniform([-1, -2], [1, 2], size=(5, 2))
        check_fixed_inputs(None, net, X, bounds, net.forward)

    def test_network_ending_in_relu(self):
        net = random_network(seed=5, relu_last=True)
        X = np.random.default_rng(6).uniform(-1, 1, size=(5, 2))
        check_fixed_inputs(None, net, X, [(-1, 1)] * 2, net.forward)

    def test_global_minimum_matches_grid(self):
        net = random_network(n_in=2, hidden=(8, 6), seed=7)
        m = pyo.ConcreteModel()
        m.a = pyo.Var(bounds=(-1, 1))
        m.b = pyo.Var(bounds=(-2, 2))
        m.nn = pyo.Block()
        add_to_pyomo(net, block=m.nn, inputs=[m.a, m.b])  # bounds from the vars
        m.obj = pyo.Objective(expr=m.nn.y[0])
        pyo.SolverFactory("highs").solve(m)

        g = np.linspace(0, 1, 401)
        grid = np.array(np.meshgrid(-1 + 2 * g, -2 + 4 * g)).reshape(2, -1).T
        assert pyo.value(m.obj) <= net.forward(grid).min() + 1e-6
        np.testing.assert_allclose(
            pyo.value(m.obj), net.forward([[m.a.value, m.b.value]])[0, 0], atol=1e-6
        )
        assert m.nn.x[0] is m.a

    def test_input_bound_errors(self):
        net = random_network()
        with pytest.raises(ValueError, match="input_bounds is required"):
            add_to_pyomo(net)
        with pytest.raises(ValueError, match="shape"):
            add_to_pyomo(net, input_bounds=[(0, 1)])
        m = pyo.ConcreteModel()
        m.a = pyo.Var(bounds=(0, 1))
        m.b = pyo.Var()
        with pytest.raises(ValueError, match="finite"):
            add_to_pyomo(net, block=m, inputs=[m.a, m.b])

    def test_variance_expression(self):
        rng = np.random.default_rng(8)
        net = random_network(seed=8)
        cov = rng.normal(size=(5, 5))
        cov = cov @ cov.T
        offset = rng.normal(size=5)
        net.std = [std_from_covariance(cov, tap=2, scale=1.5, offset=offset, d=0.3)]
        X = rng.uniform(-1, 1, size=(4, 2))
        check_fixed_inputs(None, net, X, [(-1, 1)] * 2, net.forward, net.predict_std)

        # std_from_covariance reproduces the quadratic form it factors
        _, zs = net.forward(X, return_all=True)
        z = zs[2] - offset
        expected = 1.5 * np.sqrt(np.einsum("ij,jk,ik->i", z, cov, z) + 0.3)
        np.testing.assert_allclose(net.predict_std(X)[:, 0], expected, rtol=1e-6)

    def test_include_std_adds_soc(self):
        net = random_network()
        net.std = [StdSpec(tap=-1, A=np.eye(2))]
        block = add_to_pyomo(net, input_bounds=[(-1, 1)] * 2, include_std=True)
        assert hasattr(block, "std") and hasattr(block, "std_soc")


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(40, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2
    return X, y


BOUNDS = [(-1, 1), (-1, 1)]


def untrained_dpose(X, layers=(2, 6, 8)):
    """A DPOSE with initialized (untrained) weights; export does not care."""
    import jax
    from flax import linen as nn

    from pycse.sklearn.dpose import DPOSE, _NN

    model = DPOSE(layers)
    model.nn_ = _NN(layers, nn.relu)
    model.optpars_ = model.nn_.init(jax.random.PRNGKey(0), X)
    model.calibration_factor_ = 1.3
    return model


def untrained_llpr(X, n_outputs=1, hidden_dims=(6, 5)):
    """An LLPRRegressor with initialized weights and a real covariance."""
    import jax
    from flax import linen as nn

    from pycse.sklearn.llpr_regressor import MLP, LLPRRegressor

    model = LLPRRegressor(hidden_dims=hidden_dims, activation="relu")
    model.model_ = MLP(features=(*hidden_dims, n_outputs), activation=nn.relu)
    model.params_ = model.model_.init(jax.random.PRNGKey(1), X)
    model.n_outputs_ = n_outputs
    model._compute_covariance(X)
    model.alpha_squared_ = np.linspace(0.5, 2.0, n_outputs)
    model.zeta_squared_ = np.linspace(1e-3, 1e-1, n_outputs)
    return model


@needs_highs
class TestDPOSE:
    def test_matches_predict(self, data):
        X, _ = data
        model = untrained_dpose(X)
        net = model.mip_network()
        mean, std = model.predict(X[:5], return_std=True)
        np.testing.assert_allclose(net.forward(X[:5])[:, 0], mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(net.predict_std(X[:5])[:, 0], std, rtol=1e-5)
        check_fixed_inputs(
            model,
            None,
            X[:3],
            BOUNDS,
            model.predict,
            lambda x: model.predict(x, return_std=True)[1],
        )

    def test_rejects_smooth_activation(self, data):
        from flax import linen as nn

        model = untrained_dpose(data[0])
        model.activation = nn.tanh
        with pytest.raises(ValueError, match="activation"):
            model.to_pyomo(input_bounds=BOUNDS)

    def test_unfitted(self):
        from pycse.sklearn import DPOSE

        with pytest.raises(ValueError, match="Fit"):
            DPOSE((2, 4, 4)).mip_network()


@needs_highs
class TestLLPR:
    @pytest.mark.parametrize("n_outputs", [1, 2])
    def test_matches_predict(self, data, n_outputs):
        X, _ = data
        model = untrained_llpr(X, n_outputs=n_outputs)
        net = model.mip_network()
        mean, std = model.predict(X[:5], return_std=True)
        np.testing.assert_allclose(net.forward(X[:5]), mean.reshape(5, -1), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(net.predict_std(X[:5]), std.reshape(5, -1), rtol=1e-3)
        check_fixed_inputs(
            model,
            None,
            X[:3],
            BOUNDS,
            model.predict,
            lambda x: model.predict(x, return_std=True)[1],
        )

    def test_rejects_smooth_activation(self, data):
        model = untrained_llpr(data[0])
        model.activation = "silu"
        with pytest.raises(ValueError, match="activation='relu'"):
            model.mip_network()


@needs_highs
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
class TestNNBR:
    def make(self, X, y, br, activation="relu", calibrate=False):
        from sklearn.neural_network import MLPRegressor

        from pycse.sklearn import NNBR

        nn = MLPRegressor(
            hidden_layer_sizes=(8, 6), activation=activation, max_iter=300, random_state=0
        )
        kw = {"val_X": X[:10], "val_y": y[:10]} if calibrate else {}
        return NNBR(nn, br).fit(X, y, **kw)

    @pytest.mark.parametrize("calibrate", [False, True])
    def test_bayesian_ridge(self, data, calibrate):
        from sklearn.linear_model import BayesianRidge

        X, y = data
        model = self.make(X, y, BayesianRidge(), calibrate=calibrate)
        net = model.mip_network()
        mean, std = model.predict(X[:5], return_std=True)
        np.testing.assert_allclose(net.forward(X[:5])[:, 0], mean, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(net.predict_std(X[:5])[:, 0], std, rtol=1e-6)
        check_fixed_inputs(
            model, None, X[:3], BOUNDS, model.predict, lambda x: model.predict(x, True)[1]
        )

    def test_ard_regression(self, data):
        from sklearn.linear_model import ARDRegression

        X, y = data
        model = self.make(X, y, ARDRegression())
        net = model.mip_network()
        mean, std = model.predict(X[:5], return_std=True)
        np.testing.assert_allclose(net.forward(X[:5])[:, 0], mean, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(net.predict_std(X[:5])[:, 0], std, rtol=1e-6)

    def test_rejects_smooth_activation(self, data):
        from sklearn.linear_model import BayesianRidge

        X, y = data
        model = self.make(X, y, BayesianRidge(), activation="tanh")
        with pytest.raises(ValueError, match="activation='relu'"):
            model.to_pyomo(input_bounds=BOUNDS)


@pytest.mark.slow
@needs_highs
class TestTrainedModels:
    """End-to-end with real training (slow)."""

    def test_dpose(self, data):
        from pycse.sklearn import DPOSE

        X, y = data
        model = DPOSE((2, 8, 16)).fit(X, y, maxiter=50)
        check_fixed_inputs(
            model, None, X[:3], BOUNDS, model.predict, lambda x: model.predict(x, True)[1]
        )

    def test_llpr(self, data):
        from pycse.sklearn import LLPRRegressor

        X, y = data
        model = LLPRRegressor(hidden_dims=(8, 8), activation="relu", n_epochs=20).fit(X, y)
        check_fixed_inputs(
            model,
            None,
            X[:3],
            BOUNDS,
            model.predict,
            lambda x: model.predict(x, return_std=True)[1],
        )
