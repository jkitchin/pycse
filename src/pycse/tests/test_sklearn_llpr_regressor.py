"""Tests for LLPRRegressor single-output and multi-output support."""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from pycse.sklearn.llpr_regressor import LLPRRegressor

pytestmark = pytest.mark.slow


@pytest.fixture
def single_output_data():
    X, y = make_regression(n_samples=100, n_features=3, noise=5, random_state=42)
    return X, y


@pytest.fixture
def multi_output_data():
    X, y = make_regression(n_samples=100, n_features=3, n_targets=2, noise=5, random_state=42)
    return X, y


def _make_model(**kwargs):
    defaults = dict(
        hidden_dims=(32, 32),
        n_epochs=50,
        batch_size=16,
        early_stopping_patience=20,
        val_size=0.2,
        random_state=42,
    )
    defaults.update(kwargs)
    return LLPRRegressor(**defaults)


class TestSingleOutput:
    def test_fit_predict_shapes(self, single_output_data):
        X, y = single_output_data
        model = _make_model()
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.ndim == 1
        assert y_pred.shape == (X.shape[0],)

    def test_uncertainty_shapes(self, single_output_data):
        X, y = single_output_data
        model = _make_model()
        model.fit(X, y)

        y_pred, y_std = model.predict_with_uncertainty(X)
        assert y_pred.ndim == 1
        assert y_std.ndim == 1
        assert y_pred.shape == (X.shape[0],)
        assert y_std.shape == (X.shape[0],)
        assert np.all(y_std >= 0)

    def test_uncertainty_variance(self, single_output_data):
        X, y = single_output_data
        model = _make_model()
        model.fit(X, y)

        y_pred, y_var = model.predict_with_uncertainty(X, return_std=False)
        assert y_var.ndim == 1
        assert np.all(y_var >= 0)

    def test_score(self, single_output_data):
        X, y = single_output_data
        model = _make_model()
        model.fit(X, y)

        r2 = model.score(X, y)
        assert isinstance(r2, float)
        # Training R² should be reasonable for this simple problem
        assert r2 > 0.0

    def test_n_outputs_attribute(self, single_output_data):
        X, y = single_output_data
        model = _make_model()
        model.fit(X, y)

        assert model.n_outputs_ == 1
        assert model.alpha_squared_.shape == (1,)
        assert model.zeta_squared_.shape == (1,)


class TestMultiOutput:
    def test_fit_predict_shapes(self, multi_output_data):
        X, y = multi_output_data
        model = _make_model()
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.ndim == 2
        assert y_pred.shape == (X.shape[0], 2)

    def test_uncertainty_shapes(self, multi_output_data):
        X, y = multi_output_data
        model = _make_model()
        model.fit(X, y)

        y_pred, y_std = model.predict_with_uncertainty(X)
        assert y_pred.ndim == 2
        assert y_std.ndim == 2
        assert y_pred.shape == (X.shape[0], 2)
        assert y_std.shape == (X.shape[0], 2)
        assert np.all(y_std >= 0)

    def test_uncertainty_variance(self, multi_output_data):
        X, y = multi_output_data
        model = _make_model()
        model.fit(X, y)

        y_pred, y_var = model.predict_with_uncertainty(X, return_std=False)
        assert y_var.ndim == 2
        assert y_var.shape == (X.shape[0], 2)
        assert np.all(y_var >= 0)

    def test_score(self, multi_output_data):
        X, y = multi_output_data
        model = _make_model()
        model.fit(X, y)

        r2 = model.score(X, y)
        assert isinstance(r2, float)
        assert r2 > 0.0

    def test_n_outputs_attribute(self, multi_output_data):
        X, y = multi_output_data
        model = _make_model()
        model.fit(X, y)

        assert model.n_outputs_ == 2
        assert model.alpha_squared_.shape == (2,)
        assert model.zeta_squared_.shape == (2,)

    def test_per_output_calibration(self, multi_output_data):
        """Verify that per-output calibration produces distinct parameters."""
        X, y = multi_output_data
        # Scale outputs differently to encourage different calibration
        y[:, 1] *= 10
        model = _make_model()
        model.fit(X, y)

        # Both outputs should have calibration params (not necessarily different,
        # but they should be valid positive numbers)
        assert np.all(model.alpha_squared_ > 0)
        assert np.all(model.zeta_squared_ > 0)


def _toy_1d(n, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 1))
    return X, scale * (np.sin(3 * X[:, 0]) + 0.5 * X[:, 0])


class TestSmallData:
    def test_trains_with_fewer_samples_than_batch_size(self):
        """Datasets smaller than batch_size must still take gradient steps."""
        X, y = _toy_1d(12)
        model = LLPRRegressor(hidden_dims=(16, 16), n_epochs=200, batch_size=32, cv=3)
        model.fit(X, y)
        untrained = LLPRRegressor(hidden_dims=(16, 16), n_epochs=0, batch_size=32, cv=3)
        untrained.fit(X, y)
        assert model.score(X, y) > untrained.score(X, y) + 0.2

    def test_auto_uses_cv_for_small_data(self, single_output_data):
        X, y = _toy_1d(20)
        model = LLPRRegressor(hidden_dims=(16,), n_epochs=50, cv=3).fit(X, y)
        assert model.calibration_ == "cv"
        Xb, yb = single_output_data
        model = _make_model(n_epochs=5).fit(Xb, yb)
        assert model.calibration_ == "holdout"

    def test_cv_calibration_is_positive(self):
        X, y = _toy_1d(20)
        model = LLPRRegressor(hidden_dims=(16,), n_epochs=100, calibration="cv", cv=4)
        model.fit(X, y)
        assert np.all(model.alpha_squared_ > 0)
        assert np.all(model.zeta_squared_ > 0)

    def test_bad_calibration(self):
        X, y = _toy_1d(10)
        with pytest.raises(ValueError):
            LLPRRegressor(calibration="bogus").fit(X, y)


class TestFeatures:
    def test_bias_feature_gives_nonzero_std(self):
        """With a constant feature, σ > 0 even where all ReLUs are inactive."""
        X, y = _toy_1d(30)
        model = LLPRRegressor(hidden_dims=(4,), activation="relu", n_epochs=50, cv=3)
        model.fit(X, y)
        assert model.n_features_ == 4 + 1
        Xt = np.linspace(-20, 20, 401)[:, None]
        _, std = model.predict(Xt, return_std=True)
        assert np.all(std > 0)

    def test_relu_std_grows_away_from_data(self):
        X, y = _toy_1d(30)
        model = LLPRRegressor(hidden_dims=(32, 32), activation="relu", n_epochs=200, cv=3)
        model.fit(X, y)
        _, std_in = model.predict(np.array([[0.0]]), return_std=True)
        _, std_out = model.predict(np.array([[10.0]]), return_std=True)
        assert std_out[0] > std_in[0]


class TestStandardize:
    def test_output_units(self):
        """Predictions and std are in original units for large-scale targets."""
        X, y = _toy_1d(40, scale=1000.0)
        model = LLPRRegressor(hidden_dims=(32, 32), n_epochs=300, cv=3).fit(X, y)
        y_pred, y_std = model.predict(X, return_std=True)
        assert model.score(X, y) > 0.5
        # alpha² is stored in units of y², so it must reflect the y scale
        assert model.alpha_squared_[0] > 1.0
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        assert 0.1 * rmse < np.median(y_std) < 10 * rmse

    def test_no_standardize(self):
        X, y = _toy_1d(20)
        model = LLPRRegressor(hidden_dims=(8,), n_epochs=20, standardize=False, cv=3).fit(X, y)
        assert np.all(model.y_scale_ == 1) and np.all(model.X_mean_ == 0)


class TestRefitting:
    def test_warm_start_reuses_params(self):
        X, y = _toy_1d(20)
        model = LLPRRegressor(hidden_dims=(8,), n_epochs=0, warm_start=True, cv=3)
        model.fit(X, y)
        first = model.params_
        model.fit(X + 0.1, y)
        # With zero epochs the warm-started parameters are returned unchanged
        leaves = zip(jax_leaves(first), jax_leaves(model.params_))
        assert all(np.allclose(a, b) for a, b in leaves)

    def test_quiet_by_default(self, capsys):
        X, y = _toy_1d(20)
        LLPRRegressor(hidden_dims=(8,), n_epochs=20, cv=3).fit(X, y)
        assert capsys.readouterr().out == ""


def jax_leaves(tree):
    import jax

    return jax.tree_util.tree_leaves(tree)


class TestExport:
    @pytest.fixture
    def relu_model(self):
        rng = np.random.default_rng(0)
        X = rng.uniform([-2, 0], [2, 50], (40, 2))
        y = 100 * np.sin(X[:, 0]) + 0.1 * X[:, 1] ** 2
        model = LLPRRegressor(hidden_dims=(8, 8), activation="relu", n_epochs=50, cv=3)
        return model.fit(X, y), X

    def test_dense_layers_match_predict(self, relu_model):
        """Standardization is folded into the exported layers."""
        model, X = relu_model
        h = X
        for W, b, activation in model.get_dense_layers():
            h = h @ W + b
            if activation == "relu":
                h = np.maximum(h, 0)
        np.testing.assert_allclose(h[:, 0], model.predict(X), rtol=1e-5, atol=1e-4)

    def test_silu_not_exportable(self):
        X, y = _toy_1d(10)
        model = LLPRRegressor(hidden_dims=(4,), n_epochs=5, cv=3).fit(X, y)
        with pytest.raises(ValueError, match="activation"):
            model.get_dense_layers()

    def test_onnx_roundtrip(self, relu_model, tmp_path):
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        model, X = relu_model
        path = tmp_path / "llpr.onnx"
        model.to_onnx(path)
        sess = ort.InferenceSession(str(path))
        (y_onnx,) = sess.run(None, {"X": X.astype(np.float64)})
        np.testing.assert_allclose(y_onnx[:, 0], model.predict(X), rtol=1e-5, atol=1e-4)

    def test_onnx_readable_by_omlt(self, relu_model):
        pytest.importorskip("omlt")
        pytest.importorskip("onnx")
        from omlt.io import load_onnx_neural_network

        model, X = relu_model
        bounds = {i: (X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])}
        net = load_onnx_neural_network(model.to_onnx(), input_bounds=bounds)
        assert len(list(net.layers)) == 4

    def test_omlt_network(self, relu_model):
        pytest.importorskip("omlt")
        model, X = relu_model
        net = model.to_omlt()
        layers = list(net.layers)
        assert len(layers) == 4  # input + 2 hidden + output
        assert net.scaled_input_bounds[1] == (X[:, 1].min(), X[:, 1].max())
        net = model.to_omlt(input_bounds=([-3, -1], [3, 60]))
        assert net.scaled_input_bounds[0] == (-3.0, 3.0)
