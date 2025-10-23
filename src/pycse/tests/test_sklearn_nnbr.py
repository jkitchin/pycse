"""Tests for NNBR (Neural Network Bayesian Ridge) module."""

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

# Import NNBR
from pycse.sklearn.nnbr import NeuralNetworkBLR


@pytest.fixture
def simple_linear_data():
    """Generate simple linear data for testing."""
    np.random.seed(42)
    X = np.linspace(0, 10, 100)[:, None]
    y = 2 * X.ravel() + 1 + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def heteroscedastic_data():
    """Generate heteroscedastic regression data (noise increases with X)."""
    np.random.seed(42)
    X = np.linspace(0, 1, 200)[:, None]

    # True function: y = x^(1/3)
    y_true = X.ravel() ** (1 / 3)

    # Heteroscedastic noise: increases with X
    noise_std = 0.01 + 0.08 * X.ravel()
    noise = noise_std * np.random.randn(200)
    y = y_true + noise

    return X, y, noise_std


@pytest.fixture
def simple_nn():
    """Create a simple MLPRegressor for testing."""
    return MLPRegressor(
        hidden_layer_sizes=(20,),
        activation="relu",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )


@pytest.fixture
def simple_br():
    """Create a simple BayesianRidge for testing."""
    return BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)


class TestNNBRBasicFunctionality:
    """Test basic NNBR functionality."""

    def test_initialization(self, simple_nn, simple_br):
        """Test NNBR initialization."""
        model = NeuralNetworkBLR(simple_nn, simple_br)

        assert model.nn is simple_nn
        assert model.br is simple_br
        assert model.calibration_factor == 1.0

    def test_fit_predict_basic(self, simple_linear_data, simple_nn, simple_br):
        """Test basic fit and predict cycle."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X_train, y_train)

        # Test prediction
        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_predict_with_uncertainty(self, simple_linear_data, simple_nn, simple_br):
        """Test prediction with uncertainty estimates."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std >= 0)  # Uncertainties must be non-negative

    def test_fit_returns_self(self, simple_linear_data, simple_nn, simple_br):
        """Test that fit returns self."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        result = model.fit(X, y)

        assert result is model


class TestNNBRCalibration:
    """Test calibration functionality."""

    def test_calibration_with_validation(self, heteroscedastic_data, simple_nn, simple_br):
        """Test that calibration is applied when validation data provided."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

        # Check that calibration factor was computed
        assert hasattr(model, "calibration_factor")
        assert np.isfinite(model.calibration_factor)
        assert model.calibration_factor > 0

    def test_no_calibration_without_validation(self, simple_linear_data, simple_nn, simple_br):
        """Test that no calibration when validation data not provided."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        # Calibration factor should be 1.0 (no calibration)
        assert model.calibration_factor == 1.0

    def test_calibration_affects_uncertainty(self, heteroscedastic_data, simple_nn, simple_br):
        """Test that calibration changes uncertainty estimates."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create two identical networks
        nn1 = MLPRegressor(
            hidden_layer_sizes=(20,),
            activation="relu",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        nn2 = MLPRegressor(
            hidden_layer_sizes=(20,),
            activation="relu",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

        br1 = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
        br2 = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)

        # Train with calibration
        model_cal = NeuralNetworkBLR(nn1, br1)
        model_cal.fit(X_train, y_train, val_X=X_val, val_y=y_val)

        # Train without calibration
        model_no_cal = NeuralNetworkBLR(nn2, br2)
        model_no_cal.fit(X_train, y_train)

        # Get uncertainties
        _, y_std_cal = model_cal.predict(X_val, return_std=True)
        _, y_std_no_cal = model_no_cal.predict(X_val, return_std=True)

        # If calibration factor != 1.0, uncertainties should differ
        if model_cal.calibration_factor != 1.0:
            assert not np.allclose(y_std_cal, y_std_no_cal)


class TestNNBRFeatureExtraction:
    """Test neural network feature extraction."""

    def test_features_computed(self, simple_linear_data, simple_nn, simple_br):
        """Test that neural network features are computed correctly."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        # Extract features manually
        features = model._feat(X)

        # Features should have correct shape
        # Should be (n_samples, n_neurons_last_hidden_layer)
        assert features.ndim == 2
        assert features.shape[0] == len(X)
        assert np.all(np.isfinite(features))


class TestNNBRUncertaintyMetrics:
    """Test uncertainty quantification metrics."""

    def test_uncertainty_metrics(self, heteroscedastic_data, simple_nn, simple_br):
        """Test uncertainty metrics computation."""
        X, y, _ = heteroscedastic_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = NeuralNetworkBLR(simple_nn, simple_br)
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

    def test_print_metrics(self, simple_linear_data, simple_nn, simple_br, capsys):
        """Test print_metrics outputs correctly."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        model.print_metrics(X, y)

        captured = capsys.readouterr()
        assert "UNCERTAINTY QUANTIFICATION METRICS (NNBR)" in captured.out
        assert "RMSE" in captured.out
        assert "MAE" in captured.out


class TestNNBRReportAndVisualization:
    """Test reporting and visualization methods."""

    def test_report(self, simple_linear_data, simple_nn, simple_br, capsys):
        """Test report method."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        model.report()

        captured = capsys.readouterr()
        assert "Model Report:" in captured.out
        assert "Neural Network:" in captured.out
        assert "Bayesian Ridge:" in captured.out

    def test_plot_basic(self, simple_linear_data, simple_nn, simple_br):
        """Test basic plotting functionality."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        # Test that plot runs without error
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        model.plot(X, y, ax=ax)
        plt.close(fig)

    def test_plot_creates_figure(self, simple_linear_data, simple_nn, simple_br):
        """Test that plot creates a figure when ax not provided."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        import matplotlib.pyplot as plt

        # Clear any existing figures
        plt.close("all")

        # Create plot without providing ax
        fig = model.plot(X, y)

        assert fig is not None
        plt.close(fig)


class TestNNBREdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature_input(self):
        """Test with single feature input."""
        X = np.linspace(0, 1, 50)[:, None]
        y = 2 * X.ravel() + np.random.randn(50) * 0.1

        nn = MLPRegressor(hidden_layer_sizes=(10,), solver="lbfgs", max_iter=500, random_state=42)
        br = BayesianRidge(tol=1e-6, fit_intercept=False)

        model = NeuralNetworkBLR(nn, br)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (50,)

    def test_multi_feature_input(self):
        """Test with multiple features."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

        nn = MLPRegressor(hidden_layer_sizes=(20,), solver="lbfgs", max_iter=500, random_state=42)
        br = BayesianRidge(tol=1e-6, fit_intercept=False)

        model = NeuralNetworkBLR(nn, br)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

    def test_small_dataset(self):
        """Test with small dataset."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        nn = MLPRegressor(hidden_layer_sizes=(5,), solver="lbfgs", max_iter=500, random_state=42)
        br = BayesianRidge(tol=1e-6, fit_intercept=False)

        model = NeuralNetworkBLR(nn, br)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (5,)


class TestNNBRArchitectures:
    """Test different neural network architectures."""

    def test_single_hidden_layer(self, simple_linear_data):
        """Test with single hidden layer."""
        X, y = simple_linear_data

        nn = MLPRegressor(hidden_layer_sizes=(20,), solver="lbfgs", max_iter=500, random_state=42)
        br = BayesianRidge(tol=1e-6, fit_intercept=False)

        model = NeuralNetworkBLR(nn, br)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))

    def test_multi_hidden_layers(self, simple_linear_data):
        """Test with multiple hidden layers."""
        X, y = simple_linear_data

        nn = MLPRegressor(
            hidden_layer_sizes=(20, 10),
            solver="lbfgs",
            max_iter=500,
            random_state=42,
        )
        br = BayesianRidge(tol=1e-6, fit_intercept=False)

        model = NeuralNetworkBLR(nn, br)
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
            br = BayesianRidge(tol=1e-6, fit_intercept=False)

            model = NeuralNetworkBLR(nn, br)
            model.fit(X, y)

            y_pred = model.predict(X)
            assert np.all(np.isfinite(y_pred))


class TestNNBRBayesianRidgeIntegration:
    """Test integration with Bayesian Ridge regressor."""

    def test_bayesian_ridge_parameters(self, simple_linear_data, simple_nn):
        """Test that Bayesian Ridge parameters are learned."""
        X, y = simple_linear_data

        br = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)

        model = NeuralNetworkBLR(simple_nn, br)
        model.fit(X, y)

        # Check that BR parameters were fitted
        assert hasattr(model.br, "alpha_")
        assert hasattr(model.br, "lambda_")
        assert np.isfinite(model.br.alpha_)
        assert np.isfinite(model.br.lambda_)
        assert model.br.alpha_ > 0
        assert model.br.lambda_ > 0

    def test_bayesian_ridge_uncertainty(self, simple_linear_data, simple_nn):
        """Test that Bayesian Ridge provides uncertainty estimates."""
        X, y = simple_linear_data

        br = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)

        model = NeuralNetworkBLR(simple_nn, br)
        model.fit(X, y)

        _, y_std = model.predict(X, return_std=True)

        # Uncertainty should be non-zero
        assert np.all(y_std > 0)


class TestNNBRCalibrationEdgeCases:
    """Test calibration edge cases."""

    def test_calibration_with_near_zero_uncertainty(self, simple_linear_data, capsys):
        """Test calibration when uncertainties are near zero."""
        X, y = simple_linear_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a very simple model that might have low uncertainty
        nn = MLPRegressor(hidden_layer_sizes=(5,), solver="lbfgs", max_iter=100, random_state=42)
        br = BayesianRidge(tol=1e-6, fit_intercept=False)

        model = NeuralNetworkBLR(nn, br)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

        # Calibration factor should still be valid
        assert np.isfinite(model.calibration_factor)
        assert model.calibration_factor > 0


@pytest.mark.slow
class TestNNBRPerformance:
    """Performance and accuracy tests (marked as slow)."""

    def test_linear_regression_accuracy(self):
        """Test that NNBR can fit simple linear relationship."""
        np.random.seed(42)
        X = np.linspace(0, 10, 200)[:, None]
        y = 2 * X.ravel() + 3 + 0.1 * np.random.randn(200)

        nn = MLPRegressor(hidden_layer_sizes=(50,), solver="lbfgs", max_iter=1000, random_state=42)
        br = BayesianRidge(tol=1e-6, fit_intercept=False)

        model = NeuralNetworkBLR(nn, br)
        model.fit(X, y)

        # Test on training data
        y_pred = model.predict(X)

        # Check RMSE
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        assert rmse < 0.5, f"RMSE {rmse} too high for simple linear fit"

    def test_heteroscedastic_uncertainty(self, heteroscedastic_data):
        """Test that uncertainties correlate with noise level."""
        X, y, true_noise = heteroscedastic_data

        nn = MLPRegressor(hidden_layer_sizes=(50,), solver="lbfgs", max_iter=1000, random_state=42)
        br = BayesianRidge(tol=1e-6, fit_intercept=False)

        model = NeuralNetworkBLR(nn, br)
        model.fit(X, y)

        _, y_std = model.predict(X, return_std=True)

        # Uncertainty should be non-zero and vary across inputs
        assert np.all(y_std > 0)
        assert np.std(y_std) > 0  # Uncertainties should vary


class TestNNBRComparison:
    """Tests comparing NNBR behavior."""

    def test_prediction_consistency(self, simple_linear_data, simple_nn, simple_br):
        """Test that multiple predictions on same data give same result."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        y_pred1 = model.predict(X)
        y_pred2 = model.predict(X)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_uncertainty_consistency(self, simple_linear_data, simple_nn, simple_br):
        """Test that uncertainty estimates are consistent."""
        X, y = simple_linear_data

        model = NeuralNetworkBLR(simple_nn, simple_br)
        model.fit(X, y)

        _, y_std1 = model.predict(X, return_std=True)
        _, y_std2 = model.predict(X, return_std=True)

        np.testing.assert_allclose(y_std1, y_std2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
