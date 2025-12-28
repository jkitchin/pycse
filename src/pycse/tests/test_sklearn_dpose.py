"""Tests for DPOSE (Direct Propagation of Shallow Ensembles) module."""

import numpy as np
import pytest
import jax
from sklearn.model_selection import train_test_split

# Import DPOSE
from pycse.sklearn.dpose import DPOSE


@pytest.fixture
def heteroscedastic_data():
    """Generate heteroscedastic regression data (noise increases with X)."""
    key = jax.random.PRNGKey(42)
    X = np.linspace(0, 1, 100)[:, None]

    # True function: y = x^(1/3)
    y_true = X.ravel() ** (1 / 3)

    # Heteroscedastic noise: increases with X
    noise_std = 0.01 + 0.08 * X.ravel()
    noise = noise_std * jax.random.normal(key, (100,))
    y = y_true + noise

    return X, y, noise_std


@pytest.fixture
def simple_linear_data():
    """Generate simple linear data for quick tests."""
    np.random.seed(42)
    X = np.linspace(0, 10, 50)[:, None]
    y = 2 * X.ravel() + 1 + 0.1 * np.random.randn(50)
    return X, y


class TestDPOSEBasicFunctionality:
    """Test basic DPOSE functionality."""

    def test_initialization_default(self):
        """Test DPOSE initialization with default parameters."""
        model = DPOSE(layers=(1, 20, 32))

        assert model.layers == (1, 20, 32)
        assert model.loss_type == "crps"
        assert model.optimizer == "bfgs"
        assert model.min_sigma == 1e-3
        assert model.n_ensemble == 32

    def test_initialization_custom(self):
        """Test DPOSE initialization with custom parameters."""
        model = DPOSE(
            layers=(2, 10, 16), loss_type="mse", optimizer="adam", seed=123, min_sigma=1e-4
        )

        assert model.layers == (2, 10, 16)
        assert model.loss_type == "mse"
        assert model.optimizer == "adam"
        assert model.min_sigma == 1e-4
        assert model.n_ensemble == 16

    def test_fit_predict_basic(self, simple_linear_data):
        """Test basic fit and predict cycle."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DPOSE(layers=(1, 10, 16), loss_type="mse")
        model.fit(X_train, y_train, maxiter=50)

        # Test prediction
        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_predict_with_uncertainty(self, simple_linear_data):
        """Test prediction with uncertainty estimates."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        y_pred, y_std = model.predict(X, return_std=True)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std >= 0)  # Uncertainties must be non-negative

    def test_predict_ensemble(self, simple_linear_data):
        """Test ensemble prediction."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))  # 16 ensemble members
        model.fit(X, y, maxiter=50)

        ensemble_preds = model.predict_ensemble(X)

        assert ensemble_preds.shape == (len(X), 16)
        assert np.all(np.isfinite(ensemble_preds))


class TestDPOSELossFunctions:
    """Test different loss functions."""

    def test_crps_loss(self, simple_linear_data):
        """Test CRPS loss function."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16), loss_type="crps")
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_nll_loss(self, simple_linear_data):
        """Test NLL loss function with automatic MSE pre-training."""
        X, y = simple_linear_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DPOSE(layers=(1, 10, 16), loss_type="nll")
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val, pretrain_maxiter=50, maxiter=50)

        y_pred, y_std = model.predict(X_val, return_std=True)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std > 0)  # NLL should provide uncertainty

    def test_mse_loss(self, simple_linear_data):
        """Test MSE loss function."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16), loss_type="mse")
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestDPOSEOptimizers:
    """Test different optimizers."""

    def test_bfgs_optimizer(self, simple_linear_data):
        """Test BFGS optimizer (default)."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16), optimizer="bfgs")
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_lbfgs_optimizer(self, simple_linear_data):
        """Test L-BFGS optimizer."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16), optimizer="lbfgs")
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_adam_optimizer(self, simple_linear_data):
        """Test Adam optimizer."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16), optimizer="adam")
        model.fit(X, y, maxiter=50, learning_rate=1e-3)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_sgd_optimizer(self, simple_linear_data):
        """Test SGD optimizer."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16), optimizer="sgd")
        model.fit(X, y, maxiter=50, learning_rate=1e-2)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_muon_optimizer(self, simple_linear_data):
        """Test Muon optimizer (state-of-the-art 2024)."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16), optimizer="muon")
        model.fit(X, y, maxiter=50, learning_rate=0.02)

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestDPOSECalibration:
    """Test calibration functionality."""

    def test_calibration_with_validation(self, heteroscedastic_data):
        """Test that calibration is applied when validation data provided."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DPOSE(layers=(1, 20, 32))
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=50)

        # Check that calibration factor exists
        assert hasattr(model, "calibration_factor")
        assert np.isfinite(model.calibration_factor)
        assert model.calibration_factor > 0

    def test_no_calibration_without_validation(self, simple_linear_data):
        """Test that no calibration when validation data not provided."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        # Calibration factor should be 1.0 (no calibration)
        assert model.calibration_factor == 1.0

    def test_calibration_affects_uncertainty(self, heteroscedastic_data):
        """Test that calibration changes uncertainty estimates."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train with calibration
        model_cal = DPOSE(layers=(1, 20, 32))
        model_cal.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=50)

        # Train without calibration
        model_no_cal = DPOSE(layers=(1, 20, 32), seed=19)
        model_no_cal.fit(X_train, y_train, maxiter=50)

        # Get uncertainties
        _, y_std_cal = model_cal.predict(X_val, return_std=True)
        _, y_std_no_cal = model_no_cal.predict(X_val, return_std=True)

        # If calibration factor != 1.0, uncertainties should differ
        if model_cal.calibration_factor != 1.0:
            assert not np.allclose(y_std_cal, y_std_no_cal)


class TestDPOSEUncertaintyMetrics:
    """Test uncertainty quantification metrics."""

    def test_uncertainty_metrics(self, heteroscedastic_data):
        """Test uncertainty metrics computation."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DPOSE(layers=(1, 20, 32))
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=50)

        metrics = model.uncertainty_metrics(X_val, y_val)

        # Check all expected metrics exist
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "nll" in metrics
        assert "miscalibration_area" in metrics
        assert "z_score_mean" in metrics
        assert "z_score_std" in metrics

        # Check all metrics are finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite"

    def test_print_metrics(self, simple_linear_data, capsys):
        """Test print_metrics outputs correctly."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        model.print_metrics(X, y)

        captured = capsys.readouterr()
        assert "UNCERTAINTY QUANTIFICATION METRICS" in captured.out
        assert "RMSE" in captured.out
        assert "MAE" in captured.out


class TestDPOSEReportAndVisualization:
    """Test reporting and visualization methods."""

    def test_report(self, simple_linear_data, capsys):
        """Test report method."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16), optimizer="adam", loss_type="crps")
        model.fit(X, y, maxiter=50, learning_rate=1e-3)

        model.report()

        captured = capsys.readouterr()
        assert "Optimization converged" in captured.out
        assert "adam" in captured.out
        assert "crps" in captured.out

    def test_plot_basic(self, simple_linear_data):
        """Test basic plotting functionality."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        # Test that plot runs without error
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        model.plot(X, y, ax=ax, distribution=False)
        plt.close(fig)

    def test_plot_with_distribution(self, simple_linear_data):
        """Test plotting with distribution."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        model.plot(X, y, ax=ax, distribution=True)
        plt.close(fig)


class TestDPOSEEdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature_input(self):
        """Test with single feature input."""
        X = np.linspace(0, 1, 50)[:, None]
        y = 2 * X.ravel() + np.random.randn(50) * 0.1

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert y_pred.shape == (50,)

    def test_multi_feature_input(self):
        """Test with multiple features."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

        model = DPOSE(layers=(5, 20, 32))
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

    def test_small_dataset(self):
        """Test with small dataset."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = DPOSE(layers=(1, 5, 8))
        model.fit(X, y, maxiter=50)

        y_pred = model.predict(X)
        assert y_pred.shape == (5,)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        X = np.random.randn(50, 2)
        y = np.sum(X, axis=1)

        model1 = DPOSE(layers=(2, 10, 16), seed=42)
        model1.fit(X, y, maxiter=50)
        pred1 = model1.predict(X)

        model2 = DPOSE(layers=(2, 10, 16), seed=42)
        model2.fit(X, y, maxiter=50)
        pred2 = model2.predict(X)

        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)


class TestDPOSESklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_fit_returns_self(self, simple_linear_data):
        """Test that fit returns self."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        result = model.fit(X, y, maxiter=50)

        assert result is model

    def test_attributes_exist(self):
        """Test that key attributes exist after initialization."""
        model = DPOSE(layers=(1, 20, 32), loss_type="crps", seed=42)

        assert hasattr(model, "layers")
        assert hasattr(model, "loss_type")
        assert hasattr(model, "optimizer")
        assert hasattr(model, "min_sigma")
        assert hasattr(model, "n_ensemble")
        assert hasattr(model, "calibration_factor")


class TestDPOSECallInterface:
    """Test __call__ interface."""

    def test_call_basic(self, simple_linear_data):
        """Test calling model as a function."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        # Call model directly
        y_pred = model(X)

        assert y_pred.shape == (len(X),)
        assert np.all(np.isfinite(y_pred))

    def test_call_with_std(self, simple_linear_data):
        """Test calling with return_std=True."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        y_pred, y_std = model(X, return_std=True)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)

    def test_call_with_distribution(self, simple_linear_data):
        """Test calling with distribution=True."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        ensemble = model(X, distribution=True)

        assert ensemble.shape == (len(X), 16)


class TestDPOSEUncertaintyPropagation:
    """Test uncertainty propagation for derived quantities."""

    def test_ensemble_propagation(self, simple_linear_data):
        """Test propagating uncertainty through a nonlinear function."""
        X, y = simple_linear_data

        model = DPOSE(layers=(1, 10, 16))
        model.fit(X, y, maxiter=50)

        # Get ensemble predictions
        ensemble = model.predict_ensemble(X)

        # Apply nonlinear transformation: z = exp(y)
        z_ensemble = np.exp(ensemble)

        # Get mean and std of transformed quantity
        z_mean = z_ensemble.mean(axis=1)
        z_std = z_ensemble.std(axis=1)

        assert z_mean.shape == (len(X),)
        assert z_std.shape == (len(X),)
        assert np.all(z_std >= 0)


class TestDPOSEPerformance:
    """Performance and accuracy tests."""

    def test_linear_regression_accuracy(self):
        """Test that DPOSE can fit simple linear relationship."""
        np.random.seed(42)
        X = np.linspace(0, 10, 200)[:, None]
        y = 2 * X.ravel() + 3 + 0.1 * np.random.randn(200)

        model = DPOSE(layers=(1, 20, 32), loss_type="mse")
        model.fit(X, y, maxiter=100)

        # Test on training data range
        X_test = np.array([[1], [5], [9]])
        y_pred = model.predict(X_test)
        y_expected = np.array([5, 13, 21])

        # Check that predictions are in reasonable range (RMSE < 1.5)
        rmse = np.sqrt(np.mean((y_pred - y_expected) ** 2))
        assert rmse < 1.5, f"RMSE {rmse} too high for simple linear fit"

    def test_heteroscedastic_uncertainty(self, heteroscedastic_data):
        """Test that uncertainties increase with noise level."""
        X, y, true_noise = heteroscedastic_data

        model = DPOSE(layers=(1, 20, 32))
        model.fit(X, y, maxiter=50)

        _, y_std = model.predict(X, return_std=True)

        # Uncertainty should be correlated with true noise level
        # (not perfect, but should have positive correlation)
        correlation = np.corrcoef(y_std, true_noise)[0, 1]
        assert correlation > 0.2, f"Correlation {correlation} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
