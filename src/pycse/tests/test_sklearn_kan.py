"""Tests for KAN (Kolmogorov-Arnold Networks) module."""

import numpy as np
import pytest
import jax
from sklearn.model_selection import train_test_split

from pycse.sklearn.kan import KAN


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


class TestKANBasicFunctionality:
    """Test basic KAN functionality."""

    def test_initialization_default(self):
        """Test KAN initialization with default parameters."""
        model = KAN(layers=(1, 5, 1))

        assert model.layers == (1, 5, 1)
        assert model.grid_size == 5
        assert model.spline_order == 3
        assert model.optimizer == "bfgs"
        assert model.loss_type == "mse"
        assert model.n_ensemble == 1
        assert model.n_outputs == 1

    def test_initialization_custom(self):
        """Test KAN initialization with custom parameters."""
        model = KAN(
            layers=(2, 10, 1),
            grid_size=8,
            spline_order=2,
            optimizer="adam",
            seed=123,
            loss_type="crps",
            n_ensemble=16,
        )

        assert model.layers == (2, 10, 1)
        assert model.grid_size == 8
        assert model.spline_order == 2
        assert model.optimizer == "adam"
        assert model.loss_type == "crps"
        assert model.n_ensemble == 16
        assert model.n_outputs == 1

    def test_fit_predict_basic(self, simple_linear_data):
        """Test basic fit and predict cycle."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KAN(layers=(1, 5, 1), grid_size=3)
        model.fit(X_train, y_train, maxiter=50)

        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_predict_with_uncertainty(self, simple_linear_data):
        """Test prediction with uncertainty estimates (ensemble output)."""
        X, y = simple_linear_data

        # Use n_ensemble > 1 for UQ
        model = KAN(layers=(1, 5, 1), grid_size=3, n_ensemble=16)
        model.fit(X, y, maxiter=50)

        y_pred, y_std = model.predict(X, return_std=True)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std >= 0)

    def test_predict_ensemble(self, simple_linear_data):
        """Test ensemble prediction."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 5, 1), grid_size=3, n_ensemble=16)
        model.fit(X, y, maxiter=50)

        ensemble_preds = model.predict_ensemble(X)

        assert ensemble_preds.shape == (len(X), 16)
        assert np.all(np.isfinite(ensemble_preds))


class TestKANSplineParameters:
    """Test different spline configurations."""

    def test_different_grid_sizes(self, simple_linear_data):
        """Test KAN with different grid sizes."""
        X, y = simple_linear_data

        for grid_size in [3, 5]:
            model = KAN(layers=(1, 3, 1), grid_size=grid_size)
            model.fit(X, y, maxiter=30)

            y_pred = model.predict(X)
            assert np.all(np.isfinite(y_pred))

    def test_different_spline_orders(self, simple_linear_data):
        """Test KAN with different spline orders."""
        X, y = simple_linear_data

        for order in [2, 3]:
            model = KAN(layers=(1, 3, 1), spline_order=order)
            model.fit(X, y, maxiter=50)

            y_pred = model.predict(X)
            assert np.all(np.isfinite(y_pred))


class TestKANOptimizers:
    """Test different optimizers."""

    def test_bfgs_optimizer(self, simple_linear_data):
        """Test BFGS optimizer (default)."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), optimizer="bfgs")
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_adam_optimizer(self, simple_linear_data):
        """Test Adam optimizer."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), optimizer="adam")
        model.fit(X, y, maxiter=50, learning_rate=1e-2)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestKANCalibration:
    """Test calibration functionality."""

    def test_calibration_with_validation(self, heteroscedastic_data):
        """Test that calibration is applied when validation data provided."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Use n_ensemble > 1 for UQ
        model = KAN(layers=(1, 5, 1), grid_size=3, n_ensemble=16)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=50)

        assert hasattr(model, "calibration_factor")
        assert np.isfinite(model.calibration_factor)
        assert model.calibration_factor > 0

    def test_no_calibration_without_validation(self, simple_linear_data):
        """Test that no calibration when validation data not provided."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 5, 1), grid_size=3, n_ensemble=16)
        model.fit(X, y, maxiter=50)

        assert model.calibration_factor == 1.0


class TestKANUncertaintyMetrics:
    """Test uncertainty quantification metrics."""

    def test_uncertainty_metrics_with_ensemble(self, heteroscedastic_data):
        """Test uncertainty metrics computation with ensemble output."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KAN(layers=(1, 5, 1), grid_size=3, n_ensemble=16)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=50)

        metrics = model.uncertainty_metrics(X_val, y_val)

        # Check all expected metrics exist
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "nll" in metrics
        assert "miscalibration_area" in metrics
        assert "z_score_mean" in metrics
        assert "z_score_std" in metrics

        # Check accuracy metrics are finite
        assert np.isfinite(metrics["rmse"])
        assert np.isfinite(metrics["mae"])

    def test_uncertainty_metrics_single_output(self, simple_linear_data):
        """Test uncertainty metrics with single output (no UQ)."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), grid_size=3)
        model.fit(X, y, maxiter=50)

        metrics = model.uncertainty_metrics(X, y)

        # Basic metrics should be available
        assert np.isfinite(metrics["rmse"])
        assert np.isfinite(metrics["mae"])

        # UQ metrics should be nan for single output
        assert np.isnan(metrics["nll"])


class TestKANReportAndVisualization:
    """Test reporting and visualization methods."""

    def test_report(self, simple_linear_data, capsys):
        """Test report method."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), grid_size=5, optimizer="adam")
        model.fit(X, y, maxiter=50, learning_rate=1e-2)

        model.report()

        captured = capsys.readouterr()
        assert "KAN Optimization Report" in captured.out
        assert "Grid size" in captured.out
        assert "adam" in captured.out

    def test_plot_basic(self, simple_linear_data):
        """Test basic plotting functionality."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), grid_size=3)
        model.fit(X, y, maxiter=50)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        model.plot(X, y, ax=ax)
        plt.close(fig)

    def test_plot_with_ensemble(self, simple_linear_data):
        """Test plotting with ensemble distribution."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), grid_size=3, n_ensemble=16)
        model.fit(X, y, maxiter=50)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        model.plot(X, y, ax=ax, distribution=True)
        plt.close(fig)

    def test_plot_network(self, simple_linear_data):
        """Test network graph visualization."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 4, 1), grid_size=3)
        model.fit(X, y, maxiter=50)

        import matplotlib.pyplot as plt

        fig = model.plot_network()
        assert fig is not None
        plt.close(fig)

    def test_plot_network_multi_layer(self, simple_linear_data):
        """Test network visualization with multiple hidden layers."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 2, 1), grid_size=3)
        model.fit(X, y, maxiter=50)

        import matplotlib.pyplot as plt

        fig = model.plot_network(figsize=(12, 6))
        assert fig is not None
        plt.close(fig)


class TestKANEdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature_input(self):
        """Test with single feature input."""
        np.random.seed(42)
        X = np.linspace(0, 1, 50)[:, None]
        y = 2 * X.ravel() + np.random.randn(50) * 0.1

        model = KAN(layers=(1, 3, 1), grid_size=3)
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert y_pred.shape == (50,)

    def test_multi_feature_input(self):
        """Test with multiple features."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

        model = KAN(layers=(3, 5, 1), grid_size=3)
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

    def test_small_dataset(self):
        """Test with small dataset."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)

        model = KAN(layers=(1, 3, 1), grid_size=2)
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert y_pred.shape == (5,)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.sum(X, axis=1)

        model1 = KAN(layers=(2, 3, 1), grid_size=3, seed=42)
        model1.fit(X, y, maxiter=50)
        pred1 = model1.predict(X)

        model2 = KAN(layers=(2, 3, 1), grid_size=3, seed=42)
        model2.fit(X, y, maxiter=50)
        pred2 = model2.predict(X)

        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)


class TestKANSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_fit_returns_self(self, simple_linear_data):
        """Test that fit returns self."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1))
        result = model.fit(X, y, maxiter=50)

        assert result is model

    def test_score_method(self, simple_linear_data):
        """Test score method returns R²."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KAN(layers=(1, 5, 1), grid_size=3)
        model.fit(X_train, y_train, maxiter=50)

        score = model.score(X_test, y_test)

        assert np.isfinite(score)
        # For simple linear data, R² should be reasonably high
        assert score > 0.5

    def test_attributes_exist(self):
        """Test that key attributes exist after initialization."""
        model = KAN(layers=(1, 5, 1), grid_size=5, seed=42, n_ensemble=16)

        assert hasattr(model, "layers")
        assert hasattr(model, "grid_size")
        assert hasattr(model, "spline_order")
        assert hasattr(model, "optimizer")
        assert hasattr(model, "n_ensemble")
        assert hasattr(model, "n_outputs")
        assert hasattr(model, "calibration_factor")


class TestKANCallInterface:
    """Test __call__ interface."""

    def test_call_basic(self, simple_linear_data):
        """Test calling model as a function."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), grid_size=3)
        model.fit(X, y, maxiter=50)

        y_pred = model(X)

        assert y_pred.shape == (len(X),)
        assert np.all(np.isfinite(y_pred))

    def test_call_with_std(self, simple_linear_data):
        """Test calling with return_std=True."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), grid_size=3, n_ensemble=16)
        model.fit(X, y, maxiter=50)

        y_pred, y_std = model(X, return_std=True)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)

    def test_call_with_distribution(self, simple_linear_data):
        """Test calling with distribution=True."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 3, 1), grid_size=3, n_ensemble=16)
        model.fit(X, y, maxiter=50)

        ensemble = model(X, distribution=True)

        assert ensemble.shape == (len(X), 16)


class TestKANExpressiveness:
    """Test KAN's ability to fit complex functions."""

    def test_sinusoidal_fit(self, sinusoidal_data):
        """Test that KAN can fit a sinusoidal function."""
        X, y = sinusoidal_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KAN(layers=(1, 5, 1), grid_size=5)
        model.fit(X_train, y_train, maxiter=30)

        # Check R² on test set
        score = model.score(X_test, y_test)
        assert score > 0.6, f"KAN should fit sinusoid well, got R²={score}"

    def test_polynomial_fit(self):
        """Test that KAN can fit polynomial functions."""
        np.random.seed(42)
        X = np.linspace(-2, 2, 100)[:, None]
        y = X.ravel() ** 3 - 2 * X.ravel() ** 2 + X.ravel() + 0.1 * np.random.randn(100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KAN(layers=(1, 5, 1), grid_size=5)
        model.fit(X_train, y_train, maxiter=30)

        score = model.score(X_test, y_test)
        assert score > 0.4, f"KAN should fit polynomial reasonably, got R²={score}"


class TestKANCRPSLoss:
    """Test CRPS loss for uncertainty training."""

    def test_crps_loss_trains(self, simple_linear_data):
        """Test that CRPS loss training works."""
        X, y = simple_linear_data

        model = KAN(layers=(1, 5, 1), grid_size=3, loss_type="crps", n_ensemble=16)
        model.fit(X, y, maxiter=50)

        y_pred, y_std = model.predict(X, return_std=True)

        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std > 0)


class TestKANMultiOutput:
    """Test multi-output regression support."""

    def test_multi_output_basic(self):
        """Test basic multi-output regression."""
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = np.column_stack([np.sin(2 * np.pi * X[:, 0]), np.cos(2 * np.pi * X[:, 1])])

        model = KAN(layers=(2, 5, 2), grid_size=5)
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)

        assert y_pred.shape == (100, 2)
        assert np.all(np.isfinite(y_pred))

    def test_multi_output_with_uq(self):
        """Test multi-output with uncertainty quantification."""
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = np.column_stack([X[:, 0] ** 2, X[:, 1] ** 2])

        model = KAN(layers=(2, 4, 2), grid_size=4, n_ensemble=8)
        model.fit(X, y, maxiter=50)

        mean, std = model.predict(X, return_std=True)

        assert mean.shape == (100, 2)
        assert std.shape == (100, 2)
        assert np.all(std >= 0)

    def test_multi_output_ensemble(self):
        """Test multi-output ensemble predictions."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = np.column_stack([X[:, 0], X[:, 1]])

        model = KAN(layers=(2, 3, 2), grid_size=3, n_ensemble=10)
        model.fit(X, y, maxiter=50)

        ensemble = model.predict_ensemble(X)

        assert ensemble.shape == (50, 2, 10)
        assert np.all(np.isfinite(ensemble))

    def test_multi_output_mip_export(self):
        """Test MIP export with multi-output."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]])

        model = KAN(
            layers=(2, 3, 2), spline_order=1, grid_size=4, base_activation="linear", n_ensemble=1
        )
        model.fit(X, y, maxiter=50)

        # Export to Pyomo
        model_pyomo = model.to_pyomo(input_bounds=[(0, 1), (0, 1)])

        # Check output variables exist
        assert len(list(model_pyomo.y)) == 2

    def test_multi_output_error_mismatch(self):
        """Test error when target dimensions don't match."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = np.column_stack([X[:, 0], X[:, 1], X[:, 0] + X[:, 1]])  # 3 outputs

        model = KAN(layers=(2, 3, 2))  # Expects 2 outputs

        with pytest.raises(ValueError, match="outputs"):
            model.fit(X, y, maxiter=50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
