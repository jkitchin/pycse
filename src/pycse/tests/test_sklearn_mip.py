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


# ---------------------------------------------------------------------------
# Skip connections, concave units, splines, scaling, the convex LP and trees
# ---------------------------------------------------------------------------


def randomize(params, seed=0, scale=0.7):
    """Replace every array in a parameter tree with random values of the same shape."""
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(params)
    rng = np.random.default_rng(seed)
    new = [rng.normal(size=np.shape(leaf)) * scale for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, new)


def grid_points(bounds, n=201):
    (a0, b0), (a1, b1) = bounds
    g = np.array(np.meshgrid(np.linspace(a0, b0, n), np.linspace(a1, b1, n)))
    return g.reshape(2, -1).T


def minimize(block, sense=None):
    block._min = pyo.Objective(expr=block.y[0])
    result = pyo.SolverFactory("highs").solve(block)
    assert str(result.solver.termination_condition) == "optimal"
    return pyo.value(block.y[0]), np.array([pyo.value(block.x[i]) for i in block.x])


def n_binaries(block):
    return sum(1 for v in block.component_data_objects(pyo.Var) if v.is_binary())


class TestLayers:
    def test_skip_and_concave_forward(self):
        net = MIPNetwork(
            [
                Affine([[1.0, 1.0]], [0.0, 0.0]),
                ReLU(concave=[False, True]),
                Affine([[2.0], [3.0]], [1.0], Wx=[[-1.0]]),
            ]
        )
        # x = 2: relu(2) = 2, min(0, 2) = 0 -> 2*2 + 0 - 2 + 1 = 3
        # x = -2: relu(-2) = 0, min(0, -2) = -2 -> 0 - 6 + 2 + 1 = -3
        np.testing.assert_allclose(net.forward([[2.0], [-2.0]])[:, 0], [3.0, -3.0])

    def test_pwl_clips_outside_breakpoints(self):
        from pycse.sklearn.mip import PiecewiseLinear

        layer = PiecewiseLinear(
            breakpoints=[[0.0, 1.0, 2.0]], values=[[[0.0], [1.0], [0.0]]], W=[[0.5]], b=[1.0]
        )
        net = MIPNetwork([layer])
        x = np.array([[-1.0], [0.5], [1.0], [3.0]])
        np.testing.assert_allclose(net.forward(x)[:, 0], [0.5, 1.75, 2.5, 2.5])
        with pytest.raises(ValueError, match="increasing"):
            PiecewiseLinear(breakpoints=[[0.0, 0.0]], values=np.zeros((1, 2, 1)))

    def test_bounds_sound_with_skip_concave_pwl(self):
        from pycse.sklearn.mip import PiecewiseLinear

        rng = np.random.default_rng(3)
        net = MIPNetwork(
            [
                PiecewiseLinear(
                    np.tile(np.linspace(-1, 1, 5), (2, 1)),
                    rng.normal(size=(2, 5, 4)),
                    rng.normal(size=(2, 4)),
                ),
                ReLU(concave=[True, False, True, False]),
                Affine(rng.normal(size=(4, 3)), rng.normal(size=3), Wx=rng.normal(size=(2, 3))),
                ReLU(),
                Affine(rng.normal(size=(3, 1)), [0.0]),
            ]
        )
        bounds = np.array([[-2.0, 1.5], [-0.5, 3.0]])
        X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(3000, 2))
        _, zs = net.forward(X, return_all=True)
        for z, (lo, hi) in zip(zs, propagate_bounds(net.layers, bounds)):
            assert np.all(z >= lo - 1e-9) and np.all(z <= hi + 1e-9)

    def test_scale_inputs_and_outputs(self):
        net = random_network(n_in=2, n_out=2, seed=11)
        net.layers[2].Wx = np.random.default_rng(1).normal(size=(2, 5))
        net.std = [StdSpec(tap=-1, A=np.eye(2)), StdSpec(tap=1, A=np.ones((1, 6)))]
        mean, scale = np.array([3.0, -1.0]), np.array([2.0, 0.5])
        y_mean, y_scale = np.array([10.0, -5.0]), np.array([4.0, -2.0])
        scaled = net.scale_inputs(mean, scale).scale_outputs(y_mean, y_scale)

        X = np.random.default_rng(2).normal(size=(10, 2))
        Xs = (X - mean) / scale
        np.testing.assert_allclose(scaled.forward(X), net.forward(Xs) * y_scale + y_mean)
        np.testing.assert_allclose(scaled.predict_std(X), net.predict_std(Xs) * np.abs(y_scale))


@needs_highs
class TestPyomoLayers:
    def test_skip_concave_pwl_match_forward(self):
        from pycse.sklearn.mip import PiecewiseLinear

        rng = np.random.default_rng(5)
        net = MIPNetwork(
            [
                PiecewiseLinear(
                    np.tile(np.linspace(-1, 1, 5), (2, 1)),
                    rng.normal(size=(2, 5, 4)),
                    rng.normal(size=(2, 4)),
                ),
                ReLU(concave=[True, False, True, False]),
                Affine(rng.normal(size=(4, 3)), rng.normal(size=3), Wx=rng.normal(size=(2, 3))),
                ReLU(),
                Affine(rng.normal(size=(3, 1)), [0.0]),
            ]
        )
        bounds = [(-2.0, 1.5), (-0.5, 3.0)]  # beyond the breakpoints: tests the clipping
        X = rng.uniform([-2.0, -0.5], [1.5, 3.0], size=(6, 2))
        check_fixed_inputs(None, net, X, bounds, net.forward)

        y, x = minimize(add_to_pyomo(net, input_bounds=bounds))
        assert y <= net.forward(grid_points(bounds)).min() + 1e-6
        np.testing.assert_allclose(y, net.forward(x[None, :])[0, 0], atol=1e-6)

    def test_convex_formulation_rejects_nonconvex(self):
        with pytest.raises(ValueError, match="nonnegative weights"):
            add_to_pyomo(random_network(seed=1), input_bounds=BOUNDS, formulation="convex")
        concave = MIPNetwork([Affine([[1.0]], [0.0]), ReLU(concave=[True]), Affine([[1.0]], [0])])
        with pytest.raises(ValueError, match="concave"):
            add_to_pyomo(concave, input_bounds=[(0, 1)], formulation="convex")
        with pytest.raises(ValueError, match="formulation"):
            add_to_pyomo(concave, input_bounds=[(0, 1)], formulation="lp")


# ---------------------------------------------------------------------------
# JAXICNNRegressor and JAXMonotonicRegressor
# ---------------------------------------------------------------------------


def untrained_icnn(X, y, hidden=(6, 5), nonneg_param="softplus"):
    import jax
    from sklearn.preprocessing import StandardScaler

    from pycse.sklearn.jax_icnn import JAXICNNRegressor, _init_params

    model = JAXICNNRegressor(hidden_dims=hidden, activation="relu", nonneg_param=nonneg_param)
    model.n_features_in_ = X.shape[1]
    model.scaler_X_ = StandardScaler().fit(X)
    model.scaler_y_ = StandardScaler().fit(y[:, None])
    model.params_ = randomize(_init_params(jax.random.PRNGKey(0), X.shape[1], hidden, nonneg_param))
    return model


def untrained_monotonic(X, y, convexity=None, hidden=(6, 6), nonneg_param="softplus"):
    import jax
    import jax.numpy as jnp
    from sklearn.preprocessing import StandardScaler

    from pycse.sklearn.jax_monotonic import JAXMonotonicRegressor, _forward_batch, _init_params

    model = JAXMonotonicRegressor(
        hidden_dims=hidden,
        monotonicity=[1, -1],
        activation="relu",
        nonneg_param=nonneg_param,
        convexity=convexity,
    )
    model.n_features_in_ = X.shape[1]
    model.monotonicity_ = jnp.array([1.0, -1.0])
    model.scaler_X_ = StandardScaler().fit(X)
    model.scaler_y_ = StandardScaler().fit(y[:, None])
    params = _init_params(jax.random.PRNGKey(0), 2, hidden, model.monotonicity_, nonneg_param)
    model.params_ = randomize(params, seed=3)
    _, feats = _forward_batch(
        model.params_,
        model.scaler_X_.transform(X),
        model.monotonicity_,
        "relu",
        nonneg_param,
        return_features=True,
        convexity=convexity,
    )
    feats = np.asarray(feats)
    model.cov_matrix_ = feats.T @ feats
    model.n_llpr_features_ = feats.shape[1]
    model.alpha_squared_ = 1.7
    model.zeta_squared_ = 0.05
    return model


@needs_highs
class TestICNN:
    @pytest.mark.parametrize("nonneg_param", ["softplus", "square"])
    def test_matches_predict(self, data, nonneg_param):
        X, y = data
        model = untrained_icnn(X, y, nonneg_param=nonneg_param)
        net = model.mip_network()
        Xt = np.random.default_rng(1).uniform(-1, 1, size=(8, 2))
        np.testing.assert_allclose(net.forward(Xt)[:, 0], model.predict(Xt), rtol=1e-5, atol=1e-6)
        check_fixed_inputs(model, None, Xt[:3], BOUNDS, model.predict)

    def test_convex_lp_matches_milp_and_grid(self, data):
        X, y = data
        model = untrained_icnn(X, y)
        lp = model.to_pyomo(input_bounds=BOUNDS, formulation="convex")
        assert n_binaries(lp) == 0
        y_lp, x_lp = minimize(lp)
        y_mip, _ = minimize(model.to_pyomo(input_bounds=BOUNDS))
        np.testing.assert_allclose(y_lp, y_mip, atol=1e-6)
        np.testing.assert_allclose(y_lp, model.predict(x_lp[None, :])[0], atol=1e-5)
        assert y_lp <= model.predict(grid_points(BOUNDS)).min() + 1e-5

    def test_rejects_softplus_and_mu(self, data):
        X, y = data
        model = untrained_icnn(X, y)
        model.activation = "softplus"
        with pytest.raises(ValueError, match="activation='relu'"):
            model.mip_network()
        model.activation = "relu"
        model.strong_convexity_mu = 0.1
        with pytest.raises(ValueError, match="strong_convexity_mu"):
            model.mip_network()


@needs_highs
class TestMonotonic:
    @pytest.mark.parametrize("convexity", [None, "convex", "concave"])
    def test_matches_predict(self, data, convexity):
        X, y = data
        model = untrained_monotonic(X, y, convexity=convexity)
        net = model.mip_network()
        Xt = np.random.default_rng(2).uniform(-1, 1, size=(8, 2))
        mean, std = model.predict(Xt, return_std=True)
        assert np.any(std > 0)  # not a degenerate network
        np.testing.assert_allclose(net.forward(Xt)[:, 0], mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(net.predict_std(Xt)[:, 0], std, rtol=1e-4)
        check_fixed_inputs(
            model, None, Xt[:3], BOUNDS, model.predict, lambda x: model.predict(x, True)[1]
        )

    def test_convex_lp(self, data):
        X, y = data
        model = untrained_monotonic(X, y, convexity="convex")
        lp = model.to_pyomo(input_bounds=BOUNDS, formulation="convex")
        assert n_binaries(lp) == 0 and not hasattr(lp, "variance")
        y_lp, _ = minimize(lp)
        y_mip, _ = minimize(model.to_pyomo(input_bounds=BOUNDS))
        np.testing.assert_allclose(y_lp, y_mip, atol=1e-6)

        mixed = untrained_monotonic(X, y, convexity=None)
        with pytest.raises(ValueError, match="concave"):
            mixed.to_pyomo(input_bounds=BOUNDS, formulation="convex")

    def test_rejects_softplus(self, data):
        model = untrained_monotonic(*data)
        model.activation = "softplus"
        with pytest.raises(ValueError, match="activation='relu'"):
            model.mip_network()


# ---------------------------------------------------------------------------
# KAN and KANLLPR
# ---------------------------------------------------------------------------


def untrained_kan(X, n_outputs=1, n_ensemble=1, grid_size=4):
    import jax
    import jax.numpy as jnp

    from pycse.sklearn.kan import KAN, _KANN

    model = KAN(
        layers=(2, 3, n_outputs),
        grid_size=grid_size,
        spline_order=1,
        base_activation="linear",
        n_ensemble=n_ensemble,
    )
    model.nn_ = _KANN(
        layers=(2, 3, n_outputs * n_ensemble),
        grid_size=grid_size,
        spline_order=1,
        grid_range=model.grid_range,
        base_activation="linear",
    )
    model.optpars_ = randomize(model.nn_.init(jax.random.PRNGKey(0), X), seed=4)
    model.calibration_factor_ = 1.2
    model.X_mean_ = jnp.mean(X, axis=0)
    model.X_std_ = jnp.std(X, axis=0)
    model.y_mean_ = jnp.linspace(1.0, 2.0, n_outputs)
    model.y_std_ = jnp.linspace(0.5, 3.0, n_outputs)
    return model


def untrained_kanllpr(X, n_outputs=1, grid_size=4):
    import jax
    import jax.numpy as jnp

    from pycse.sklearn.kan_llpr import KANLLPR, _KANNWithFeatures

    model = KANLLPR(
        layers=(2, 3, n_outputs), grid_size=grid_size, spline_order=1, base_activation="linear"
    )
    model.nn_ = _KANNWithFeatures(
        layers=(2, 3, n_outputs),
        grid_size=grid_size,
        spline_order=1,
        grid_range=model.grid_range,
        base_activation="linear",
    )
    model.optpars_ = randomize(model.nn_.init(jax.random.PRNGKey(0), X), seed=5)
    model.X_mean_ = jnp.mean(X, axis=0)
    model.X_std_ = jnp.std(X, axis=0)
    model.y_mean_ = jnp.linspace(1.0, 2.0, n_outputs)
    model.y_std_ = jnp.linspace(0.5, 3.0, n_outputs)
    model._compute_covariance(X)
    model.alpha_squared_ = jnp.linspace(0.5, 2.0, n_outputs)
    model.zeta_squared_ = jnp.linspace(1e-3, 1e-1, n_outputs)
    return model


@needs_highs
class TestKAN:
    @pytest.mark.parametrize("n_outputs,n_ensemble", [(1, 1), (2, 1), (2, 3)])
    def test_matches_predict(self, data, n_outputs, n_ensemble):
        X, _ = data
        model = untrained_kan(X, n_outputs, n_ensemble)
        net = model.mip_network()
        # Inputs beyond the training range land outside the spline grid (clipping).
        Xt = np.random.default_rng(3).uniform(-1.5, 1.5, size=(8, 2))
        mean, std = model.predict(Xt, return_std=True)
        np.testing.assert_allclose(net.forward(Xt), np.reshape(mean, (8, -1)), rtol=1e-5, atol=1e-5)
        if n_ensemble > 1:
            np.testing.assert_allclose(net.predict_std(Xt), np.reshape(std, (8, -1)), rtol=1e-4)
        else:
            assert net.std is None
        bounds = [(-1.5, 1.5)] * 2
        std_fn = (lambda x: model.predict(x, return_std=True)[1]) if n_ensemble > 1 else None
        check_fixed_inputs(model, None, Xt[:3], bounds, model.predict, std_fn)

    def test_global_minimum_and_binaries(self, data):
        X, _ = data
        model = untrained_kan(X, grid_size=4)
        block = model.to_pyomo(input_bounds=BOUNDS)
        # One set of segment binaries per input of each layer, not per edge.
        assert n_binaries(block) <= (2 + 3) * (4 + 2)
        y, x = minimize(block)
        assert y <= model.predict(grid_points(BOUNDS)).min() + 1e-5
        np.testing.assert_allclose(y, model.predict(x[None, :])[0], atol=1e-5)

    def test_default_bounds_cover_the_grid(self, data):
        X, _ = data
        model = untrained_kan(X)
        block = model.to_pyomo()
        expected = np.asarray(model.X_mean_) + np.outer(
            model.grid_range, np.asarray(model.X_std_) + 1e-8
        )
        np.testing.assert_allclose([block.x[i].bounds for i in range(2)], expected.T)

    def test_rejects_cubic_and_silu(self, data):
        X, _ = data
        model = untrained_kan(X)
        model.spline_order = 3
        with pytest.raises(ValueError, match="spline_order=1"):
            model.mip_network()
        model.spline_order = 1
        model.base_activation = "silu"
        with pytest.raises(ValueError, match="base_activation='linear'"):
            model.to_pyomo(input_bounds=BOUNDS)


@needs_highs
class TestKANLLPR:
    @pytest.mark.parametrize("n_outputs", [1, 2])
    def test_matches_predict(self, data, n_outputs):
        X, _ = data
        model = untrained_kanllpr(X, n_outputs)
        net = model.mip_network()
        Xt = np.random.default_rng(4).uniform(-1.5, 1.5, size=(8, 2))
        mean, std = model.predict(Xt, return_std=True)
        np.testing.assert_allclose(net.forward(Xt), np.reshape(mean, (8, -1)), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(net.predict_std(Xt), np.reshape(std, (8, -1)), rtol=1e-3)
        check_fixed_inputs(
            model,
            None,
            Xt[:3],
            [(-1.5, 1.5)] * 2,
            model.predict,
            lambda x: model.predict(x, return_std=True)[1],
        )


# ---------------------------------------------------------------------------
# LeafModelRegressor and LinearRegressionUQ
# ---------------------------------------------------------------------------


@needs_highs
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestLeafModel:
    def make(self, data, leaf_model=None, **kw):
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from pycse.sklearn import LeafModelRegressor

        if leaf_model is None:
            leaf_model = make_pipeline(StandardScaler(), LinearRegression())
        X, y = data
        return LeafModelRegressor(leaf_model, min_samples_leaf=6, random_state=0, **kw).fit(X, y)

    @pytest.mark.parametrize("refine", [False, True])
    def test_matches_predict(self, data, refine):
        model = self.make(data, refine_splits=refine)
        tree = model.mip_network()
        assert len(tree.leaves) == model.get_n_leaves()
        Xt = np.random.default_rng(5).uniform(-1, 1, size=(50, 2))
        np.testing.assert_allclose(tree.forward(Xt)[:, 0], model.predict(Xt), atol=1e-9)
        check_fixed_inputs(model, None, Xt[:4], BOUNDS, model.predict)

    def test_global_minimum(self, data):
        model = self.make(data)
        y, x = minimize(model.to_pyomo(input_bounds=BOUNDS))
        grid = grid_points(BOUNDS, n=301)
        assert y <= model.predict(grid).min() + 1e-6
        # The optimum may sit on a split, where the tree picks one side; the MIP
        # value must match one of the neighboring leaves.
        eps = np.array([[0, 0], [1e-7, 0], [-1e-7, 0], [0, 1e-7], [0, -1e-7]])
        near = np.clip(x + eps, -1, 1)
        assert np.min(np.abs(model.predict(near) - y)) < 1e-5

    def test_smaller_bounds_drop_leaves(self, data):
        model = self.make(data)
        full = model.to_pyomo(input_bounds=BOUNDS)
        small = model.to_pyomo(input_bounds=[(0.0, 0.2), (0.0, 0.2)])
        assert n_binaries(small) < n_binaries(full)

    def test_rejects_nonlinear_leaf(self, data):
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        model = self.make(data, make_pipeline(PolynomialFeatures(2), LinearRegression()))
        with pytest.raises(ValueError, match="not affine"):
            model.mip_network()


@needs_highs
class TestLinearRegressionUQ:
    def test_matches_predict(self, data):
        from pycse.sklearn import LinearRegressionUQ

        X, y = data
        Xa = np.column_stack([np.ones(len(X)), X])
        model = LinearRegressionUQ().fit(Xa, y)
        net = model.mip_network()
        mean, std = model.predict(Xa[:5], return_std=True)
        np.testing.assert_allclose(net.forward(Xa[:5])[:, 0], mean, rtol=1e-10)
        np.testing.assert_allclose(net.predict_std(Xa[:5])[:, 0], std, rtol=1e-8)
        block = model.to_pyomo(input_bounds=[(1, 1), (-1, 1), (-1, 1)])
        assert n_binaries(block) == 0
        y_fixed, var = solve_at(block, Xa[0])
        np.testing.assert_allclose(y_fixed, mean[:1], rtol=1e-8)
        np.testing.assert_allclose(np.sqrt(var), std[:1], rtol=1e-6)


@pytest.mark.slow
@needs_highs
class TestTrainedPiecewiseModels:
    """End-to-end with real training (slow)."""

    def test_icnn(self, data):
        from pycse.sklearn import JAXICNNRegressor

        X, y = data
        model = JAXICNNRegressor(hidden_dims=(8, 8), activation="relu", epochs=5).fit(X, y)
        check_fixed_inputs(model, None, X[:3], BOUNDS, model.predict)

    def test_monotonic(self, data):
        from pycse.sklearn import JAXMonotonicRegressor

        X, y = data
        model = JAXMonotonicRegressor(
            hidden_dims=(8, 8), monotonicity=[1, 0], activation="relu", epochs=5
        ).fit(X, y)
        check_fixed_inputs(
            model, None, X[:3], BOUNDS, model.predict, lambda x: model.predict(x, True)[1]
        )

    def test_kan(self, data):
        from pycse.sklearn import KAN

        X, y = data
        model = KAN(layers=(2, 3, 1), spline_order=1, base_activation="linear", grid_size=4)
        model.fit(X, y, maxiter=20)
        check_fixed_inputs(model, None, X[:3], BOUNDS, model.predict)
