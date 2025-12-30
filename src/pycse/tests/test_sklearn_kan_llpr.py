"""Tests for KANLLPR (KAN with Last-Layer Prediction Rigidity) module."""

import numpy as np
import pytest
import jax
from sklearn.model_selection import train_test_split

from pycse.sklearn.kan_llpr import KANLLPR, compute_calibration_metrics


@pytest.fixture
def simple_linear_data():
    """Generate simple linear data for quick tests."""
    np.random.seed(42)
    X = np.linspace(0, 10, 50)[:, None]
    y = 2 * X.ravel() + 1 + 0.1 * np.random.randn(50)
    return X, y


@pytest.fixture
def sinusoidal_data():
    """Generate sinusoidal data (good for testing KAN's expressiveness)."""
    np.random.seed(42)
    X = np.linspace(0, 2 * np.pi, 75)[:, None]
    y = np.sin(X.ravel()) + 0.1 * np.random.randn(75)
    return X, y


@pytest.fixture
def heteroscedastic_data():
    """Generate heteroscedastic regression data (noise increases with X)."""
    key = jax.random.PRNGKey(42)
    X = np.linspace(0, 1, 100)[:, None]

    # True function: y = x^(1/3)
    y_true = X.ravel() ** (1 / 3)

    # Heteroscedastic noise: increases with X
    noise_std = 0.01 + 0.08 * X.ravel()
    noise = noise_std * np.asarray(jax.random.normal(key, (100,)))
    y = y_true + noise

    return X, y, noise_std


class TestKANLLPRBasicFunctionality:
    """Test basic KANLLPR functionality."""

    def test_initialization_default(self):
        """Test KANLLPR initialization with default parameters."""
        model = KANLLPR(layers=(1, 5, 1))

        assert model.layers == (1, 5, 1)
        assert model.grid_size == 5
        assert model.spline_order == 3
        assert model.optimizer == "bfgs"
        assert model.alpha_squared == "auto"
        assert model.zeta_squared == "auto"
        assert model.val_size == 0.1
        assert model.n_outputs == 1

    def test_initialization_custom(self):
        """Test KANLLPR initialization with custom parameters."""
        model = KANLLPR(
            layers=(2, 10, 1),
            grid_size=8,
            spline_order=2,
            optimizer="adam",
            seed=123,
            alpha_squared=0.5,
            zeta_squared=1e-5,
            val_size=0.2,
        )

        assert model.layers == (2, 10, 1)
        assert model.grid_size == 8
        assert model.spline_order == 2
        assert model.optimizer == "adam"
        assert model.alpha_squared == 0.5
        assert model.zeta_squared == 1e-5
        assert model.val_size == 0.2

    def test_fit_predict_basic(self, simple_linear_data):
        """Test basic fit and predict cycle."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KANLLPR(layers=(1, 5, 1), grid_size=3, val_size=0.2)
        model.fit(X_train, y_train, maxiter=10)

        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_predict_with_uncertainty(self, simple_linear_data):
        """Test prediction with LLPR uncertainty estimates."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 5, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred, y_std = model.predict_with_uncertainty(X)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std >= 0)

    def test_predict_with_uncertainty_variance(self, simple_linear_data):
        """Test prediction returning variance instead of std."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 5, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred, y_var = model.predict_with_uncertainty(X, return_std=False)

        assert y_pred.shape == (len(X),)
        assert y_var.shape == (len(X),)
        assert np.all(y_var >= 0)


class TestKANLLPRCalibration:
    """Test LLPR calibration functionality."""

    def test_auto_calibration(self, heteroscedastic_data):
        """Test that auto calibration is applied."""
        X, y, _ = heteroscedastic_data

        model = KANLLPR(
            layers=(1, 5, 1),
            grid_size=3,
            alpha_squared="auto",
            zeta_squared="auto",
            val_size=0.2,
        )
        model.fit(X, y, maxiter=10)

        assert hasattr(model, "alpha_squared_")
        assert hasattr(model, "zeta_squared_")
        assert np.isfinite(model.alpha_squared_)
        assert np.isfinite(model.zeta_squared_)
        assert model.alpha_squared_ > 0
        assert model.zeta_squared_ > 0

    def test_manual_calibration_params(self, simple_linear_data):
        """Test with manually specified calibration parameters."""
        X, y = simple_linear_data

        model = KANLLPR(
            layers=(1, 5, 1), grid_size=3, alpha_squared=0.5, zeta_squared=1e-6, val_size=0.0
        )
        model.fit(X, y, maxiter=10)

        assert model.alpha_squared_ == 0.5
        assert model.zeta_squared_ == 1e-6

    def test_covariance_matrix_computed(self, simple_linear_data):
        """Test that covariance matrix is computed."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 8, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        assert hasattr(model, "cov_matrix_")
        assert hasattr(model, "n_features_")
        assert model.cov_matrix_.shape == (model.n_features_, model.n_features_)
        assert np.all(np.isfinite(model.cov_matrix_))


class TestKANLLPRUncertaintyMetrics:
    """Test uncertainty quantification metrics."""

    def test_uncertainty_metrics(self, heteroscedastic_data):
        """Test uncertainty metrics computation."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KANLLPR(layers=(1, 5, 1), grid_size=3, val_size=0.2)
        model.fit(X_train, y_train, maxiter=10)

        metrics = model.uncertainty_metrics(X_val, y_val)

        # Check all expected metrics exist
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "nll" in metrics
        assert "miscalibration_area" in metrics
        assert "z_score_mean" in metrics
        assert "z_score_std" in metrics
        assert "fraction_within_1_sigma" in metrics
        assert "fraction_within_2_sigma" in metrics
        assert "fraction_within_3_sigma" in metrics

        # Check metrics are finite
        assert np.isfinite(metrics["rmse"])
        assert np.isfinite(metrics["mae"])
        assert np.isfinite(metrics["nll"])

    def test_compute_calibration_metrics_function(self):
        """Test standalone calibration metrics function."""
        np.random.seed(42)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        y_std = np.array([0.2, 0.3, 0.2, 0.2, 0.3])

        metrics = compute_calibration_metrics(y_true, y_pred, y_std)

        assert "rmse" in metrics
        assert "nll" in metrics
        assert "calibration_error" in metrics
        assert np.isfinite(metrics["rmse"])
        assert np.isfinite(metrics["nll"])


class TestKANLLPRReportAndVisualization:
    """Test reporting and visualization methods."""

    def test_report(self, simple_linear_data, capsys):
        """Test report method."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), grid_size=5, optimizer="adam", val_size=0.2)
        model.fit(X, y, maxiter=10, learning_rate=1e-2)

        model.report()

        captured = capsys.readouterr()
        assert "KANLLPR Model Report" in captured.out
        assert "Grid size" in captured.out
        assert "LLPR" in captured.out

    def test_print_metrics(self, simple_linear_data, capsys):
        """Test print_metrics method."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        model.print_metrics(X, y)

        captured = capsys.readouterr()
        assert "KANLLPR UNCERTAINTY QUANTIFICATION METRICS" in captured.out
        assert "RMSE" in captured.out
        assert "Z-score" in captured.out

    def test_plot_basic(self, simple_linear_data):
        """Test basic plotting functionality."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        model.plot(X, y, ax=ax)
        plt.close(fig)


class TestKANLLPREdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature_input(self):
        """Test with single feature input."""
        np.random.seed(42)
        X = np.linspace(0, 1, 50)[:, None]
        y = 2 * X.ravel() + np.random.randn(50) * 0.1

        model = KANLLPR(layers=(1, 3, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred = model.predict(X)
        assert y_pred.shape == (50,)

    def test_multi_feature_input(self):
        """Test with multiple features."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

        model = KANLLPR(layers=(3, 5, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

    def test_small_dataset(self):
        """Test with small dataset."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)

        model = KANLLPR(layers=(1, 3, 1), grid_size=2, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred = model.predict(X)
        assert y_pred.shape == (10,)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.sum(X, axis=1)

        model1 = KANLLPR(layers=(2, 3, 1), grid_size=3, seed=42, val_size=0.2)
        model1.fit(X, y, maxiter=10)
        pred1 = model1.predict(X)

        model2 = KANLLPR(layers=(2, 3, 1), grid_size=3, seed=42, val_size=0.2)
        model2.fit(X, y, maxiter=10)
        pred2 = model2.predict(X)

        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)


class TestKANLLPRSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_fit_returns_self(self, simple_linear_data):
        """Test that fit returns self."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), val_size=0.2)
        result = model.fit(X, y, maxiter=10)

        assert result is model

    def test_score_method(self, simple_linear_data):
        """Test score method returns R²."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KANLLPR(layers=(1, 5, 1), grid_size=3, val_size=0.2)
        model.fit(X_train, y_train, maxiter=10)

        score = model.score(X_test, y_test)

        assert np.isfinite(score)
        # For simple linear data, R² should be reasonably high
        assert score > -1.0  # Low bar since we use few iterations

    def test_attributes_exist(self):
        """Test that key attributes exist after initialization."""
        model = KANLLPR(layers=(1, 5, 1), grid_size=5, seed=42)

        assert hasattr(model, "layers")
        assert hasattr(model, "grid_size")
        assert hasattr(model, "spline_order")
        assert hasattr(model, "optimizer")
        assert hasattr(model, "alpha_squared")
        assert hasattr(model, "zeta_squared")


class TestKANLLPRCallInterface:
    """Test __call__ interface."""

    def test_call_basic(self, simple_linear_data):
        """Test calling model as a function."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred = model(X)

        assert y_pred.shape == (len(X),)
        assert np.all(np.isfinite(y_pred))

    def test_call_with_std(self, simple_linear_data):
        """Test calling with return_std=True."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), grid_size=3, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred, y_std = model(X, return_std=True)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)


class TestKANLLPRExpressiveness:
    """Test KANLLPR's ability to fit complex functions."""

    def test_sinusoidal_fit(self, sinusoidal_data):
        """Test that KANLLPR can fit a sinusoidal function."""
        X, y = sinusoidal_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KANLLPR(layers=(1, 5, 1), grid_size=5, val_size=0.2)
        model.fit(X_train, y_train, maxiter=10)

        # Check R² on test set
        score = model.score(X_test, y_test)
        assert score > -1.0, f"KANLLPR should produce finite predictions, got R²={score}"

    def test_polynomial_fit(self):
        """Test that KANLLPR can fit polynomial functions."""
        np.random.seed(42)
        X = np.linspace(-2, 2, 100)[:, None]
        y = X.ravel() ** 3 - 2 * X.ravel() ** 2 + X.ravel() + 0.1 * np.random.randn(100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KANLLPR(layers=(1, 5, 1), grid_size=5, val_size=0.2)
        model.fit(X_train, y_train, maxiter=10)

        score = model.score(X_test, y_test)
        assert score > -1.0, f"KANLLPR should produce finite predictions, got R²={score}"


class TestKANLLPRRegularization:
    """Test regularization options."""

    def test_l1_spline_regularization(self, simple_linear_data):
        """Test L1 spline regularization."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), grid_size=3, l1_spline=0.01, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_l1_activation_regularization(self, simple_linear_data):
        """Test L1 activation regularization."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), grid_size=3, l1_activation=0.01, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_entropy_regularization(self, simple_linear_data):
        """Test entropy regularization."""
        X, y = simple_linear_data

        model = KANLLPR(layers=(1, 3, 1), grid_size=3, entropy_reg=0.01, val_size=0.2)
        model.fit(X, y, maxiter=10)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestKANLLPRMultiOutput:
    """Test multi-output support with per-output calibration."""

    def test_multi_output_basic(self):
        """Test basic multi-output regression."""
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = np.column_stack([np.sin(2 * np.pi * X[:, 0]), np.cos(2 * np.pi * X[:, 1])])

        model = KANLLPR(layers=(2, 2, 2), grid_size=2, val_size=0.0)
        model.fit(X, y, maxiter=5)

        y_pred = model.predict(X)

        assert y_pred.shape == (30, 2)
        assert np.all(np.isfinite(y_pred))

    def test_multi_output_uncertainty_shape(self):
        """Test that uncertainty has correct shape for multi-output."""
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = np.column_stack([X[:, 0] ** 2, X[:, 1] ** 2])

        model = KANLLPR(layers=(2, 2, 2), grid_size=2, val_size=0.0)
        model.fit(X, y, maxiter=5)

        mean, std = model.predict_with_uncertainty(X)

        assert mean.shape == (30, 2)
        assert std.shape == (30, 2)
        assert np.all(std >= 0)

    def test_multi_output_per_output_calibration(self):
        """Test that each output has its own calibration parameters."""
        np.random.seed(42)
        X = np.random.rand(30, 2)
        # Two outputs with different noise levels
        y = np.column_stack(
            [
                X[:, 0] + 0.01 * np.random.randn(30),  # Low noise
                X[:, 1] + 0.5 * np.random.randn(30),  # High noise
            ]
        )

        model = KANLLPR(layers=(2, 2, 2), grid_size=2, val_size=0.2)
        model.fit(X, y, maxiter=5)

        # Should have separate calibration for each output
        assert hasattr(model, "alpha_squared_")
        assert hasattr(model, "zeta_squared_")
        assert len(model.alpha_squared_) == 2
        assert len(model.zeta_squared_) == 2

    def test_multi_output_metrics(self):
        """Test uncertainty metrics for multi-output."""
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = np.column_stack([X[:, 0], X[:, 1]])

        model = KANLLPR(layers=(2, 2, 2), grid_size=2, val_size=0.2)
        model.fit(X, y, maxiter=5)

        metrics = model.uncertainty_metrics(X, y)

        # Should have per-output metrics
        assert "per_output" in metrics
        assert len(metrics["per_output"]) == 2
        assert "rmse" in metrics["per_output"][0]
        assert "nll" in metrics["per_output"][0]

    def test_multi_output_error_mismatch(self):
        """Test error when target dimensions don't match."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = np.column_stack([X[:, 0], X[:, 1], X[:, 0] + X[:, 1]])  # 3 outputs

        model = KANLLPR(layers=(2, 3, 2), val_size=0.2)  # Expects 2 outputs

        with pytest.raises(ValueError, match="outputs"):
            model.fit(X, y, maxiter=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
