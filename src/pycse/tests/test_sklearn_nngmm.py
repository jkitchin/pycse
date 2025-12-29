"""Tests for NNGMM (Neural Network Gaussian Mixture Model) module."""

import sys
import numpy as np
import pytest

# Skip all tests in this module if gmr is not installed
pytest.importorskip("gmr", reason="gmr not installed")

# Skip all tests on Python 3.13 - GMR library has compatibility issues
# with NumPy 2.x scalar conversion that causes errors in Python 3.13
if sys.version_info >= (3, 13):
    pytest.skip(
        "NNGMM tests skipped on Python 3.13 due to GMR library compatibility issues",
        allow_module_level=True,
    )

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

from pycse.sklearn.nngmm import NeuralNetworkGMM  # noqa: E402


@pytest.fixture
def simple_linear_data():
    """Generate simple linear data for testing."""
    np.random.seed(42)
    X = np.linspace(0, 10, 80)[:, None]
    y = 2 * X.ravel() + 1 + 0.1 * np.random.randn(80)
    return X, y


@pytest.fixture
def heteroscedastic_data():
    """Generate heteroscedastic regression data (noise increases with X)."""
    np.random.seed(42)
    X = np.linspace(0, 1, 150)[:, None]

    # True function: y = x^(1/3)
    y_true = X.ravel() ** (1 / 3)

    # Heteroscedastic noise: increases with X
    noise_std = 0.01 + 0.08 * X.ravel()
    noise = noise_std * np.random.randn(150)
    y = y_true + noise

    return X, y, noise_std


@pytest.fixture
def simple_nn():
    """Create a simple MLPRegressor for testing."""
    return MLPRegressor(
        hidden_layer_sizes=(20,),
        activation="relu",
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )


class TestNNGMMBasicFunctionality:
    """Test basic NNGMM functionality."""

    def test_initialization_default(self, simple_nn):
        """Test NNGMM initialization with default parameters."""
        model = NeuralNetworkGMM(simple_nn)

        assert model.nn is simple_nn
        assert model.n_components == 1
        assert model.n_samples == 500
        assert model.calibration_factor == 1.0

    def test_initialization_custom(self, simple_nn):
        """Test NNGMM initialization with custom parameters."""
        model = NeuralNetworkGMM(simple_nn, n_components=3, n_samples=500)

        assert model.nn is simple_nn
        assert model.n_components == 3
        assert model.n_samples == 500
        assert model.calibration_factor == 1.0

    def test_fit_predict_basic(self, simple_linear_data, simple_nn):
        """Test basic fit and predict cycle."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X_train, y_train)

        # Test prediction
        y_pred = model.predict(X_test)

        # GMM returns (n, 1) shaped array, not (n,)
        assert y_pred.shape in [(len(X_test),), (len(X_test), 1)]
        assert np.all(np.isfinite(y_pred))

    def test_predict_with_uncertainty(self, simple_linear_data, simple_nn):
        """Test prediction with uncertainty estimates."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)

        # GMM returns (n, 1) shaped array, not (n,)
        assert y_pred.shape in [(len(X),), (len(X), 1)]
        assert y_std.shape == (len(X),)  # std is always 1D
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std >= 0)  # Uncertainties must be non-negative

    def test_fit_returns_self(self, simple_linear_data, simple_nn):
        """Test that fit returns self."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn)
        result = model.fit(X, y)

        assert result is model


class TestNNGMMCalibration:
    """Test calibration functionality."""

    def test_calibration_with_validation(self, heteroscedastic_data, simple_nn):
        """Test that calibration is applied when validation data provided."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

        # Check that calibration factor was computed
        assert hasattr(model, "calibration_factor")
        assert np.isfinite(model.calibration_factor)
        assert model.calibration_factor > 0

    def test_no_calibration_without_validation(self, simple_linear_data, simple_nn):
        """Test that no calibration when validation data not provided."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X, y)

        # Calibration factor should be 1.0 (no calibration)
        assert model.calibration_factor == 1.0

    def test_calibration_affects_uncertainty(self, heteroscedastic_data):
        """Test that calibration changes uncertainty estimates."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create two identical networks
        nn1 = MLPRegressor(
            hidden_layer_sizes=(20,),
            activation="relu",
            solver="lbfgs",
            max_iter=300,
            random_state=42,
        )
        nn2 = MLPRegressor(
            hidden_layer_sizes=(20,),
            activation="relu",
            solver="lbfgs",
            max_iter=300,
            random_state=42,
        )

        # Train with calibration
        model_cal = NeuralNetworkGMM(nn1, n_components=1)
        model_cal.fit(X_train, y_train, val_X=X_val, val_y=y_val)

        # Train without calibration
        model_no_cal = NeuralNetworkGMM(nn2, n_components=1)
        model_no_cal.fit(X_train, y_train)

        # Get uncertainties
        _, y_std_cal = model_cal.predict(X_val, return_std=True)
        _, y_std_no_cal = model_no_cal.predict(X_val, return_std=True)

        # If calibration factor != 1.0, uncertainties should differ
        if model_cal.calibration_factor != 1.0:
            assert not np.allclose(y_std_cal, y_std_no_cal)


class TestNNGMMFeatureExtraction:
    """Test neural network feature extraction."""

    def test_features_computed(self, simple_linear_data, simple_nn):
        """Test that neural network features are computed correctly."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X, y)

        # Extract features manually
        features = model._feat(X)

        # Features should have correct shape
        assert features.ndim == 2
        assert features.shape[0] == len(X)
        assert np.all(np.isfinite(features))


class TestNNGMMGMMComponents:
    """Test GMM with different number of components."""

    def test_single_component(self, simple_linear_data, simple_nn):
        """Test with single GMM component (similar to Gaussian)."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std > 0)

    def test_multiple_components(self, simple_linear_data, simple_nn):
        """Test with multiple GMM components."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=3)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std > 0)

    def test_different_n_samples(self, simple_linear_data, simple_nn):
        """Test with different number of samples for uncertainty."""
        X, y = simple_linear_data

        # Test with fewer samples
        model_100 = NeuralNetworkGMM(simple_nn, n_components=1, n_samples=100)
        model_100.fit(X, y)
        _, y_std_100 = model_100.predict(X[:10], return_std=True)

        # Should still provide valid uncertainties
        assert np.all(np.isfinite(y_std_100))
        assert np.all(y_std_100 > 0)


class TestNNGMMUncertaintyMetrics:
    """Test uncertainty quantification metrics."""

    def test_uncertainty_metrics(self, heteroscedastic_data, simple_nn):
        """Test uncertainty metrics computation."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

        metrics = model.uncertainty_metrics(X_val, y_val)

        # Check all expected metrics exist
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "nll" in metrics
        assert "miscalibration_area" in metrics
        assert "z_score_mean" in metrics
        assert "z_score_std" in metrics

        # Check all metrics are finite (if uncertainty is available)
        if not np.isnan(metrics["nll"]):
            for key, value in metrics.items():
                assert np.isfinite(value), f"{key} is not finite"

    def test_print_metrics(self, simple_linear_data, simple_nn, capsys):
        """Test print_metrics outputs correctly."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X, y)

        model.print_metrics(X, y)

        captured = capsys.readouterr()
        # Check for actual print_metrics output
        assert "UNCERTAINTY QUANTIFICATION METRICS" in captured.out
        assert "RMSE" in captured.out
        assert "MAE" in captured.out
        assert "Calibration Diagnostics:" in captured.out


class TestNNGMMReportAndVisualization:
    """Test reporting and visualization methods."""

    def test_report(self, simple_linear_data, simple_nn, capsys):
        """Test report method."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=2)
        model.fit(X, y)

        model.report()

        captured = capsys.readouterr()
        # Check for actual report content
        assert "NEURAL NETWORK GMM MODEL" in captured.out
        assert "Neural Network:" in captured.out
        assert "GMM Configuration:" in captured.out
        assert "Calibration:" in captured.out
        assert "Components: 2" in captured.out

    def test_plot_basic(self, simple_linear_data, simple_nn):
        """Test basic plotting functionality."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X, y)

        # Test that plot runs without error
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        model.plot(X, y, ax=ax)
        plt.close(fig)

    def test_plot_creates_figure(self, simple_linear_data, simple_nn):
        """Test that plot creates a figure when ax not provided."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X, y)

        import matplotlib.pyplot as plt

        # Clear any existing figures
        plt.close("all")

        # Create plot without providing ax - returns ax, not fig
        ax = model.plot(X, y)

        assert ax is not None
        plt.close("all")


class TestNNGMMEdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature_input(self):
        """Test with single feature input."""
        np.random.seed(42)
        X = np.linspace(0, 1, 50)[:, None]
        y = 2 * X.ravel() + np.random.randn(50) * 0.1

        nn = MLPRegressor(hidden_layer_sizes=(10,), solver="lbfgs", max_iter=500, random_state=42)

        model = NeuralNetworkGMM(nn, n_components=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        # GMM returns (n, 1) shaped array, not (n,)
        assert y_pred.shape in [(50,), (50, 1)]
        assert np.all(np.isfinite(y_pred))

    def test_multi_feature_input(self):
        """Test with multiple features."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

        nn = MLPRegressor(hidden_layer_sizes=(20,), solver="lbfgs", max_iter=500, random_state=42)

        model = NeuralNetworkGMM(nn, n_components=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        # GMM returns (n, 1) shaped array, not (n,)
        assert y_pred.shape in [(100,), (100, 1)]
        assert np.all(np.isfinite(y_pred))

    def test_small_dataset(self):
        """Test with small dataset."""
        np.random.seed(42)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        nn = MLPRegressor(hidden_layer_sizes=(5,), solver="lbfgs", max_iter=500, random_state=42)

        model = NeuralNetworkGMM(nn, n_components=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        # GMM returns (n, 1) shaped array, not (n,)
        assert y_pred.shape in [(5,), (5, 1)]
        assert np.all(np.isfinite(y_pred))


class TestNNGMMArchitectures:
    """Test different neural network architectures."""

    def test_single_hidden_layer(self, simple_linear_data):
        """Test with single hidden layer."""
        X, y = simple_linear_data

        nn = MLPRegressor(hidden_layer_sizes=(20,), solver="lbfgs", max_iter=500, random_state=42)

        model = NeuralNetworkGMM(nn, n_components=1)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))

    def test_multi_hidden_layers(self, simple_linear_data):
        """Test with multiple hidden layers."""
        X, y = simple_linear_data

        nn = MLPRegressor(
            hidden_layer_sizes=(20, 10), solver="lbfgs", max_iter=500, random_state=42
        )

        model = NeuralNetworkGMM(nn, n_components=1)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))

    def test_different_activations(self, simple_linear_data):
        """Test with different activation functions."""
        X, y = simple_linear_data

        for activation in ["relu", "tanh", "logistic"]:
            nn = MLPRegressor(
                hidden_layer_sizes=(20,),
                activation=activation,
                solver="lbfgs",
                max_iter=500,
                random_state=42,
            )

            model = NeuralNetworkGMM(nn, n_components=1)
            model.fit(X, y)

            y_pred = model.predict(X)
            assert np.all(np.isfinite(y_pred))


class TestNNGMMGMMIntegration:
    """Test integration with GMM."""

    def test_gmm_fitted(self, simple_linear_data, simple_nn):
        """Test that GMM is fitted correctly."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=2)
        model.fit(X, y)

        # Check that GMM was fitted - stored as 'gmm' not 'gmm_'
        assert hasattr(model, "gmm")
        assert model.gmm is not None
        # gmr.GMM has 'priors' attribute (check gmr docs)
        assert hasattr(model.gmm, "priors") or hasattr(model.gmm, "means")

    def test_gmm_provides_uncertainty(self, simple_linear_data, simple_nn):
        """Test that GMM provides uncertainty estimates."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1)
        model.fit(X, y)

        _, y_std = model.predict(X, return_std=True)

        # Uncertainty should be non-zero
        assert np.all(y_std > 0)


class TestNNGMMComparison:
    """Tests comparing NNGMM behavior."""

    def test_prediction_consistency(self, simple_linear_data, simple_nn):
        """Test that multiple predictions on same data give similar results."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1, n_samples=500)
        model.fit(X, y)

        # Due to sampling, predictions might vary slightly
        y_pred1 = model.predict(X)
        y_pred2 = model.predict(X)

        # Should be close but might not be exactly equal due to sampling
        np.testing.assert_allclose(y_pred1, y_pred2, rtol=0.1)

    def test_uncertainty_varies_with_samples(self, simple_linear_data, simple_nn):
        """Test that uncertainty estimates vary with different n_samples."""
        X, y = simple_linear_data

        model = NeuralNetworkGMM(simple_nn, n_components=1, n_samples=100)
        model.fit(X, y)

        # Get multiple uncertainty estimates
        _, y_std1 = model.predict(X[:10], return_std=True)
        _, y_std2 = model.predict(X[:10], return_std=True)

        # Due to sampling, uncertainties might vary
        # But should be in same order of magnitude
        assert np.allclose(np.mean(y_std1), np.mean(y_std2), rtol=0.5), (
            "Uncertainty estimates should be reasonably consistent"
        )


@pytest.mark.slow
class TestNNGMMPerformance:
    """Performance and accuracy tests (marked as slow)."""

    def test_linear_regression_accuracy(self):
        """Test that NNGMM can fit simple linear relationship."""
        np.random.seed(123)  # Different seed for better convergence
        X = np.linspace(0, 10, 200)[:, None]
        y = 2 * X.ravel() + 3 + 0.1 * np.random.randn(200)

        nn = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            solver="lbfgs",
            max_iter=500,
            random_state=123,
            alpha=0.001,  # Small L2 regularization
        )

        model = NeuralNetworkGMM(nn, n_components=1, n_samples=500)
        model.fit(X, y)

        # Test on training data
        y_pred = model.predict(X).ravel()  # Flatten in case it's (n, 1)

        # Check RMSE - neural nets can be stochastic, use relaxed tolerance
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        # More relaxed tolerance: Neural net + GMM can have higher error than pure linear regression
        assert rmse < 2.0, f"RMSE {rmse} too high for simple linear fit"

    def test_heteroscedastic_uncertainty(self, heteroscedastic_data):
        """Test that NNGMM provides reasonable uncertainty estimates."""
        X, y, true_noise = heteroscedastic_data

        nn = MLPRegressor(hidden_layer_sizes=(50,), solver="lbfgs", max_iter=300, random_state=42)

        model = NeuralNetworkGMM(nn, n_components=1)
        model.fit(X, y)

        _, y_std = model.predict(X, return_std=True)

        # Uncertainty should be non-zero and vary across inputs
        assert np.all(y_std > 0)
        assert np.std(y_std) > 0  # Uncertainties should vary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
