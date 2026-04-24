"""
Tests to verify recent changes to ZENN.

Tests:
1. ZENNRegressor with NLL default loss
2. ZENNClassifier with get_calibration_metrics
3. Basic sklearn compatibility
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest


class TestZENNRegressorNLL:
    """Test ZENNRegressor with NLL loss (now default)."""

    def test_nll_is_default(self):
        """Verify NLL is the default loss type."""
        from pycse.sklearn.zenn import ZENNRegressor

        reg = ZENNRegressor()
        assert reg.loss_type == "nll", f"Expected 'nll' default, got '{reg.loss_type}'"

    @pytest.mark.slow
    def test_nll_fit_predict(self):
        """Test basic fit/predict with NLL loss."""
        from pycse.sklearn.zenn import ZENNRegressor

        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(50) * 0.1

        reg = ZENNRegressor(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=100,
            loss_type="nll",
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        y_pred = reg.predict(X)
        assert y_pred.shape == (50,), f"Expected shape (50,), got {y_pred.shape}"

        # Check reasonable predictions (R² > 0.5)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5, f"R² too low: {r2:.3f}"

    @pytest.mark.slow
    def test_calibration_metrics(self):
        """Test get_calibration_metrics method."""
        from pycse.sklearn.zenn import ZENNRegressor

        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        noise_std = 0.2
        y = X.flatten() ** 2 + np.random.randn(50) * noise_std

        reg = ZENNRegressor(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=200,
            loss_type="nll",
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        metrics = reg.get_calibration_metrics(X, y)

        assert "empirical_std" in metrics
        assert "learned_std" in metrics
        assert "calibration_ratio" in metrics
        assert "coverage_95" in metrics

        # Check values are reasonable
        assert metrics["empirical_std"] > 0
        assert metrics["learned_std"] > 0
        assert 0 <= metrics["coverage_95"] <= 1

    def test_mse_loss_still_works(self):
        """Verify MSE loss option still works."""
        from pycse.sklearn.zenn import ZENNRegressor

        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = X.flatten() ** 2

        reg = ZENNRegressor(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=100,
            loss_type="mse",  # Explicit MSE
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        y_pred = reg.predict(X)
        assert y_pred.shape == (50,)


class TestZENNClassifierCalibration:
    """Test ZENNClassifier calibration methods."""

    def test_get_calibration_metrics_exists(self):
        """Verify get_calibration_metrics method exists."""
        from pycse.sklearn.zenn import ZENNClassifier

        clf = ZENNClassifier()
        assert hasattr(clf, "get_calibration_metrics")

    @pytest.mark.slow
    def test_calibration_metrics_output(self):
        """Test get_calibration_metrics returns expected keys."""
        from pycse.sklearn.zenn import ZENNClassifier
        from sklearn.datasets import make_classification

        np.random.seed(42)
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42
        )

        clf = ZENNClassifier(
            n_temperatures=1, hidden_dims=(16, 16), max_epochs=50, random_state=42, verbose=0
        )
        clf.fit(X, y)

        metrics = clf.get_calibration_metrics(X, y)

        # Check all expected keys exist
        assert "accuracy" in metrics
        assert "mean_confidence" in metrics
        assert "ece" in metrics
        assert "mce" in metrics
        assert "overconfidence" in metrics

        # Check values are reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["mean_confidence"] <= 1
        assert 0 <= metrics["ece"] <= 1
        assert 0 <= metrics["mce"] <= 1

    def test_calibration_curve_exists(self):
        """Verify calibration_curve method exists."""
        from pycse.sklearn.zenn import ZENNClassifier

        clf = ZENNClassifier()
        assert hasattr(clf, "calibration_curve")


class TestSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_regressor_sklearn_api(self):
        """Test ZENNRegressor follows sklearn API."""
        from pycse.sklearn.zenn import ZENNRegressor
        from sklearn.base import BaseEstimator, RegressorMixin

        reg = ZENNRegressor()

        # Check inheritance from sklearn base classes
        assert isinstance(reg, BaseEstimator), "Should inherit from BaseEstimator"
        assert isinstance(reg, RegressorMixin), "Should inherit from RegressorMixin"

        # Check required methods
        assert hasattr(reg, "fit")
        assert hasattr(reg, "predict")
        assert hasattr(reg, "score")
        assert hasattr(reg, "get_params")
        assert hasattr(reg, "set_params")

    def test_classifier_sklearn_api(self):
        """Test ZENNClassifier follows sklearn API."""
        from pycse.sklearn.zenn import ZENNClassifier
        from sklearn.base import BaseEstimator, ClassifierMixin

        clf = ZENNClassifier()

        # Check inheritance from sklearn base classes
        assert isinstance(clf, BaseEstimator), "Should inherit from BaseEstimator"
        assert isinstance(clf, ClassifierMixin), "Should inherit from ClassifierMixin"

        # Check required methods
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")
        assert hasattr(clf, "score")
        assert hasattr(clf, "get_params")
        assert hasattr(clf, "set_params")


class TestUncertaintyDecomposition:
    """Test uncertainty quantification features."""

    @pytest.mark.slow
    def test_regressor_uncertainty(self):
        """Test regressor uncertainty decomposition."""
        from pycse.sklearn.zenn import ZENNRegressor

        np.random.seed(42)
        X = np.linspace(-2, 2, 30).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(30) * 0.1

        reg = ZENNRegressor(
            n_configs=4, hidden_dims=(8, 8), max_epochs=100, random_state=42, verbose=0
        )
        reg.fit(X, y)

        result = reg.predict_with_uncertainty(X)

        assert "prediction" in result
        assert "epistemic" in result
        assert "aleatoric" in result
        assert "total" in result

        # Check shapes
        assert result["prediction"].shape == (30,)
        assert result["epistemic"].shape == (30,)
        assert result["aleatoric"].shape == (30,)

    @pytest.mark.slow
    def test_classifier_uncertainty(self):
        """Test classifier uncertainty decomposition."""
        from pycse.sklearn.zenn import ZENNClassifier
        from sklearn.datasets import make_classification

        np.random.seed(42)
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=3, n_informative=5, random_state=42
        )

        clf = ZENNClassifier(
            n_temperatures=1, hidden_dims=(16, 16), max_epochs=50, random_state=42, verbose=0
        )
        clf.fit(X, y)

        result = clf.predict_with_uncertainty(X, return_decomposition=True)

        assert "predictions" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert "epistemic" in result
        assert "aleatoric" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
