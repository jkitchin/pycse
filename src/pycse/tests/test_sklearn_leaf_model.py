"""Tests for LeafModelRegressor module."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

from pycse.sklearn.leaf_model import LeafModelRegressor


@pytest.fixture
def simple_linear_data():
    """Generate simple linear data."""
    np.random.seed(42)
    X = np.linspace(0, 10, 100)[:, None]
    y = 2 * X.ravel() + 1 + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def piecewise_data():
    """Generate piecewise data (different slopes in different regions)."""
    np.random.seed(42)
    X = np.linspace(0, 10, 200)[:, None]
    y = np.where(X.ravel() < 5, 2 * X.ravel() + 1, 5 * X.ravel() - 14)
    y = y + 0.2 * np.random.randn(200)
    return X, y


@pytest.fixture
def nonlinear_data():
    """Generate nonlinear data (exponential-like)."""
    np.random.seed(42)
    X = np.linspace(0, 2, 100)[:, None]
    y = np.exp(X.ravel()) + 0.1 * np.random.randn(100)
    return X, y


class TestLeafModelBasicFunctionality:
    """Test basic functionality of LeafModelRegressor."""

    def test_initialization(self):
        """Test that model initializes correctly."""
        leaf_model = LinearRegression()
        model = LeafModelRegressor(leaf_model=leaf_model, min_samples_leaf=5)

        assert model.leaf_model is leaf_model
        assert model.min_samples_leaf == 5

    def test_fit_predict(self, simple_linear_data):
        """Test basic fit and predict."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        y_pred = model.predict(X)

        assert y_pred.shape == y.shape
        assert np.all(np.isfinite(y_pred))

    def test_predict_with_return_std(self, simple_linear_data):
        """Test predict with uncertainties."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=BayesianRidge(), min_samples_leaf=5)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)

        assert y_pred.shape == y.shape
        assert y_std.shape == y.shape
        assert np.all(y_std > 0)
        assert np.all(np.isfinite(y_std))

    def test_score_method(self, simple_linear_data):
        """Test score method returns R²."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        score = model.score(X, y)

        assert isinstance(score, float)
        assert 0 <= score <= 1.0  # R² for this data should be good


class TestLeafModelCalibration:
    """Test calibration functionality."""

    def test_calibration_with_validation(self, piecewise_data):
        """Test that calibration is applied when validation data provided."""
        X, y = piecewise_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LeafModelRegressor(leaf_model=BayesianRidge(), min_samples_leaf=10)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

        assert hasattr(model, "calibration_factor_")
        assert np.isfinite(model.calibration_factor_)
        assert model.calibration_factor_ > 0

    def test_calibration_affects_uncertainty(self, piecewise_data):
        """Test that calibration changes uncertainty estimates."""
        X, y = piecewise_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Without calibration
        model1 = LeafModelRegressor(leaf_model=BayesianRidge(), min_samples_leaf=10)
        model1.fit(X_train, y_train)
        _, y_std1 = model1.predict(X_val, return_std=True)

        # With calibration
        model2 = LeafModelRegressor(leaf_model=BayesianRidge(), min_samples_leaf=10)
        model2.fit(X_train, y_train, val_X=X_val, val_y=y_val)
        _, y_std2 = model2.predict(X_val, return_std=True)

        # Calibration should change uncertainties
        assert not np.allclose(y_std1, y_std2)


class TestLeafModelUncertaintyQuantification:
    """Test uncertainty quantification features."""

    def test_residual_fallback_uq(self, simple_linear_data):
        """Test fallback to residual-based UQ when model doesn't support return_std."""
        X, y = simple_linear_data

        # LinearRegression doesn't support return_std
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)

        # Should still get uncertainties via fallback
        assert np.all(np.isfinite(y_std))
        assert np.all(y_std > 0)

    def test_uncertainty_metrics(self, piecewise_data):
        """Test uncertainty_metrics returns proper dict."""
        X, y = piecewise_data
        model = LeafModelRegressor(leaf_model=BayesianRidge(), min_samples_leaf=10)
        model.fit(X, y)

        metrics = model.uncertainty_metrics(X, y)

        # Check that all expected keys are present
        expected_keys = [
            "rmse",
            "mae",
            "r2",
            "mean_abs_z_score",
            "std_z_score",
            "nll",
            "within_1std",
            "within_2std",
            "within_3std",
            "miscalibration_area",
            "mean_uncertainty",
            "std_uncertainty",
        ]

        for key in expected_keys:
            assert key in metrics
            assert np.isfinite(metrics[key])


class TestLeafModelExtrapolation:
    """Test extrapolation warnings."""

    def test_extrapolation_warning(self, simple_linear_data):
        """Test that extrapolation triggers warning."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        # Extrapolate beyond training data
        X_extrap = np.array([[15.0], [20.0]])

        with pytest.warns(UserWarning, match="outside training data bounds"):
            model.predict(X_extrap, warn_extrapolation=True)

    def test_no_warning_when_disabled(self, simple_linear_data):
        """Test that extrapolation warning can be disabled."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        X_extrap = np.array([[15.0], [20.0]])

        # Should not raise warning when disabled
        model.predict(X_extrap, warn_extrapolation=False)


class TestLeafModelSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_get_params(self):
        """Test get_params returns leaf_model."""
        leaf_model = LinearRegression()
        model = LeafModelRegressor(leaf_model=leaf_model, min_samples_leaf=5)

        params = model.get_params(deep=False)

        assert "leaf_model" in params
        assert params["leaf_model"] is leaf_model
        # Note: min_samples_leaf is a DecisionTreeRegressor param, should be there
        # but might be in different format depending on sklearn version

    def test_get_params_deep(self):
        """Test get_params with deep=True includes leaf_model params."""
        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
        model = LeafModelRegressor(leaf_model=pipe, min_samples_leaf=5)

        params = model.get_params(deep=True)

        # Should have nested parameters
        assert any(key.startswith("leaf_model__") for key in params.keys())

    def test_set_params(self):
        """Test set_params modifies leaf_model."""
        lr1 = LinearRegression(fit_intercept=True)
        lr2 = LinearRegression(fit_intercept=False)
        model = LeafModelRegressor(leaf_model=lr1, min_samples_leaf=5)

        # Test setting leaf_model parameter
        model.set_params(leaf_model=lr2)

        assert model.leaf_model is lr2

    def test_set_params_nested(self):
        """Test set_params with nested leaf_model parameters."""
        lr = LinearRegression(fit_intercept=True)
        model = LeafModelRegressor(leaf_model=lr, min_samples_leaf=5)

        model.set_params(leaf_model__fit_intercept=False)

        assert model.leaf_model.fit_intercept is False


class TestLeafModelDiagnostics:
    """Test diagnostic methods."""

    def test_report_runs(self, simple_linear_data, capsys):
        """Test that report() runs without error."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        model.report()

        captured = capsys.readouterr()
        assert "LeafModelRegressor Summary" in captured.out
        assert "Number of leaves" in captured.out

    def test_print_metrics_runs(self, simple_linear_data, capsys):
        """Test that print_metrics() runs without error."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=BayesianRidge(), min_samples_leaf=5)
        model.fit(X, y)

        model.print_metrics(X, y)

        captured = capsys.readouterr()
        assert "Uncertainty Quantification Metrics" in captured.out
        assert "R² Score" in captured.out

    @pytest.mark.skip(reason="Requires matplotlib and interactive display")
    def test_plot_runs(self, simple_linear_data):
        """Test that plot() runs without error."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        # This would show a plot in interactive mode
        model.plot(X, y)


class TestLeafModelEdgeCases:
    """Test edge cases and error handling."""

    def test_small_leaf_warning(self):
        """Test warning when leaf has few samples."""
        np.random.seed(42)
        X = np.array([[0], [1], [10], [11]])
        y = np.array([0, 1, 10, 11])

        model = LeafModelRegressor(leaf_model=LinearRegression(), max_depth=2, min_samples_leaf=1)

        with pytest.warns(UserWarning, match="only .* sample"):
            model.fit(X, y)

    def test_failed_leaf_model_fitting(self):
        """Test handling when leaf model fails to fit."""
        np.random.seed(42)
        # Create data that will cause rank deficiency
        X = np.array([[1], [1], [1], [2], [2], [2], [10], [10], [10], [11], [11], [11]])
        y = np.array([1, 1, 1, 2, 2, 2, 10, 10, 10, 11, 11, 11])

        # Use polynomial with degree higher than possible rank
        pipe = Pipeline([("poly", PolynomialFeatures(degree=15)), ("lr", LinearRegression())])

        # Force very small leaves with deep tree
        model = LeafModelRegressor(leaf_model=pipe, max_depth=4, min_samples_leaf=1)

        # Fit - may or may not warn depending on data distribution
        model.fit(X, y)

        # Should still be able to predict even if some leaves failed
        y_pred = model.predict(X[:5])
        assert np.all(np.isfinite(y_pred))

    def test_single_sample_data(self):
        """Test with very small dataset."""
        X = np.array([[1.0]])
        y = np.array([2.0])

        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=1)

        with pytest.warns(UserWarning):
            model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == y.shape


class TestLeafModelPiecewiseModeling:
    """Test that leaf model correctly captures piecewise patterns."""

    def test_piecewise_linear_accuracy(self, piecewise_data):
        """Test that model can fit piecewise linear data well."""
        X, y = piecewise_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=20, max_depth=3)
        model.fit(X, y)

        y_pred = model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        # Should fit piecewise data reasonably well
        assert rmse < 1.0

    def test_nonlinear_with_polynomial(self, nonlinear_data):
        """Test using polynomial features in leaf models."""
        X, y = nonlinear_data

        pipe = Pipeline([("poly", PolynomialFeatures(degree=2)), ("lr", LinearRegression())])

        model = LeafModelRegressor(leaf_model=pipe, min_samples_leaf=10, max_depth=3)
        model.fit(X, y)

        score = model.score(X, y)

        # Should fit nonlinear data reasonably well with polynomials
        assert score > 0.8


class TestLeafModelMemoryEfficiency:
    """Test memory efficiency improvements."""

    def test_training_data_not_stored(self, simple_linear_data):
        """Test that full training data is not stored after fitting."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        # Old version stored xtrain, ytrain
        assert not hasattr(model, "xtrain")
        assert not hasattr(model, "ytrain")

        # New version stores only statistics
        assert hasattr(model, "leaf_stats_")
        assert hasattr(model, "X_min_")
        assert hasattr(model, "X_max_")

    def test_leaf_stats_structure(self, simple_linear_data):
        """Test that leaf_stats contains expected keys."""
        X, y = simple_linear_data
        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=5)
        model.fit(X, y)

        for leaf_id, stats in model.leaf_stats_.items():
            assert "n_samples" in stats
            assert "X_mean" in stats
            assert "X_std" in stats
            assert "y_mean" in stats
            assert "y_std" in stats
            assert "residual_std" in stats


@pytest.mark.slow
class TestLeafModelPerformance:
    """Performance tests (marked as slow)."""

    def test_complex_piecewise_fit(self):
        """Test fitting complex piecewise function."""
        np.random.seed(42)
        X = np.linspace(0, 10, 500)[:, None]

        # Multi-segment piecewise function
        y = np.piecewise(
            X.ravel(),
            [
                X.ravel() < 2.5,
                (X.ravel() >= 2.5) & (X.ravel() < 5),
                (X.ravel() >= 5) & (X.ravel() < 7.5),
                X.ravel() >= 7.5,
            ],
            [lambda x: x, lambda x: 2 * x - 2.5, lambda x: -x + 12.5, lambda x: 0.5 * x + 1.25],
        )
        y = y + 0.2 * np.random.randn(500)

        model = LeafModelRegressor(leaf_model=LinearRegression(), min_samples_leaf=30, max_depth=4)
        model.fit(X, y)

        score = model.score(X, y)
        assert score > 0.85

    def test_bayesian_ridge_integration(self):
        """Test integration with Bayesian Ridge for UQ."""
        np.random.seed(42)
        X = np.linspace(0, 10, 300)[:, None]

        # Heteroscedastic noise
        noise_std = 0.1 + 0.5 * (X.ravel() / 10)
        y = 2 * np.sin(X.ravel()) + np.random.randn(300) * noise_std

        pipe = Pipeline([("poly", PolynomialFeatures(degree=3)), ("br", BayesianRidge())])

        model = LeafModelRegressor(leaf_model=pipe, min_samples_leaf=40, max_depth=3)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

        metrics = model.uncertainty_metrics(X_val, y_val)

        # Calibrated uncertainties should be reasonable
        assert 0.5 < metrics["std_z_score"] < 1.5
        assert metrics["within_1std"] > 0.5
        assert metrics["within_2std"] > 0.8
