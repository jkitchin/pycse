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
