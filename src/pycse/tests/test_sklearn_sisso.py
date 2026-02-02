"""Tests for SISSO (Sure Independence Screening and Sparsifying Operator) module."""

import numpy as np
import pytest

# Mark all tests in this module as slow (SISSO training involves optimization)
pytestmark = pytest.mark.slow


@pytest.fixture
def simple_additive_data():
    """Generate simple additive data: y = x0 + x1."""
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = X[:, 0] + X[:, 1] + 0.05 * np.random.randn(50)
    return X, y


@pytest.fixture
def multiplicative_data():
    """Generate multiplicative data: y = x0 * x1."""
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = X[:, 0] * X[:, 1] + 0.05 * np.random.randn(50)
    return X, y


@pytest.fixture
def noisy_linear_data():
    """Generate noisy linear data with known noise level."""
    np.random.seed(42)
    X = np.random.rand(60, 2)
    noise = 0.1 * np.random.randn(60)
    y = 2 * X[:, 0] + 3 * X[:, 1] + noise
    return X, y, noise


class TestSISSOBasicFunctionality:
    """Test basic SISSO functionality."""

    def test_initialization_default(self):
        """Test SISSO initialization with default parameters."""
        from pycse.sklearn.sisso import SISSO

        model = SISSO()

        assert model.operators is None
        assert model.n_expansion == 2
        assert model.n_term == 2
        assert model.k == 20
        assert model.use_gpu is False
        assert model.feature_names is None

    def test_initialization_custom(self):
        """Test SISSO initialization with custom parameters."""
        from pycse.sklearn.sisso import SISSO

        model = SISSO(
            operators=["+", "-", "*"],
            n_expansion=3,
            n_term=1,
            k=15,
            use_gpu=False,
            feature_names=["a", "b"],
        )

        assert model.operators == ["+", "-", "*"]
        assert model.n_expansion == 3
        assert model.n_term == 1
        assert model.k == 15
        assert model.feature_names == ["a", "b"]

    def test_fit_returns_self(self, simple_additive_data):
        """fit() should return self for method chaining."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=1)
        result = model.fit(X, y)
        assert result is model

    def test_predict_shape(self, simple_additive_data):
        """predict() returns correct shape."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=1).fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (50,)

    def test_equation_attribute(self, simple_additive_data):
        """equation_ should be a string after fitting."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=1).fit(X, y)
        assert isinstance(model.equation_, str)
        assert len(model.equation_) > 0

    def test_rmse_attribute(self, simple_additive_data):
        """rmse_ should be set after fitting."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=1).fit(X, y)
        assert hasattr(model, "rmse_")
        assert model.rmse_ >= 0

    def test_r2_attribute(self, simple_additive_data):
        """r2_ should be set after fitting."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=1).fit(X, y)
        assert hasattr(model, "r2_")
        assert model.r2_ <= 1.0


class TestSISSOPrediction:
    """Test SISSO prediction functionality."""

    def test_predict_not_fitted(self):
        """predict() should raise error if not fitted."""
        from pycse.sklearn.sisso import SISSO

        model = SISSO()
        X = np.random.rand(10, 2)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_predict_finite_values(self, simple_additive_data):
        """Predictions should be finite."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=1).fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_predict_reasonable_accuracy(self, simple_additive_data):
        """Predictions should have reasonable accuracy on simple data."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=2).fit(X, y)
        y_pred = model.predict(X)

        # RMSE should be relatively small for simple additive function
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        assert rmse < 0.5  # Generous bound

    def test_score_method(self, simple_additive_data):
        """score() should return R² value."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=2).fit(X, y)
        score = model.score(X, y)

        assert isinstance(score, float)
        assert score <= 1.0
        # For simple additive data, should have good fit
        assert score > 0.5


class TestSISSOUncertainty:
    """Test SISSO uncertainty quantification."""

    def test_return_std_shape(self, noisy_linear_data):
        """return_std=True should return (y_pred, y_std)."""
        from pycse.sklearn.sisso import SISSO

        X, y, _ = noisy_linear_data
        model = SISSO(n_expansion=1, n_term=2).fit(X, y)

        y_pred, y_std = model.predict(X, return_std=True)
        assert y_pred.shape == (60,)
        assert y_std.shape == (60,)

    def test_uncertainty_positive(self, noisy_linear_data):
        """Uncertainties should be positive."""
        from pycse.sklearn.sisso import SISSO

        X, y, _ = noisy_linear_data
        model = SISSO(n_expansion=1, n_term=2).fit(X, y)

        _, y_std = model.predict(X, return_std=True)
        assert np.all(y_std > 0)

    def test_uncertainty_finite(self, noisy_linear_data):
        """Uncertainties should be finite."""
        from pycse.sklearn.sisso import SISSO

        X, y, _ = noisy_linear_data
        model = SISSO(n_expansion=1, n_term=2).fit(X, y)

        _, y_std = model.predict(X, return_std=True)
        assert np.all(np.isfinite(y_std))

    def test_sigma_computed(self, noisy_linear_data):
        """sigma_ should be computed after fit."""
        from pycse.sklearn.sisso import SISSO

        X, y, _ = noisy_linear_data
        model = SISSO(n_expansion=1, n_term=2).fit(X, y)

        assert hasattr(model, "sigma_")
        assert model.sigma_ > 0

    def test_calibration_factor_computed(self, noisy_linear_data):
        """calibration_factor_ should be computed after fit."""
        from pycse.sklearn.sisso import SISSO

        X, y, _ = noisy_linear_data
        model = SISSO(n_expansion=1, n_term=2).fit(X, y)

        assert hasattr(model, "calibration_factor_")
        assert model.calibration_factor_ > 0

    def test_uncertainty_increases_extrapolation(self, simple_additive_data):
        """Uncertainty should increase for extrapolation."""
        from pycse.sklearn.sisso import SISSO

        X_train, y_train = simple_additive_data
        model = SISSO(n_expansion=1, n_term=2).fit(X_train, y_train)

        # In-distribution points
        X_in = np.random.rand(20, 2)  # Same [0, 1] range
        _, std_in = model.predict(X_in, return_std=True)

        # Extrapolation points (outside [0, 1] range)
        X_out = np.random.rand(20, 2) + 5  # [5, 6] range
        _, std_out = model.predict(X_out, return_std=True)

        # Extrapolation uncertainty should be larger on average
        assert np.mean(std_out) > np.mean(std_in)


class TestSISSOSklearnCompatibility:
    """Test sklearn compatibility."""

    def test_sklearn_clone(self, simple_additive_data):
        """Model should be clonable."""
        from sklearn.base import clone
        from pycse.sklearn.sisso import SISSO

        model = SISSO(n_expansion=2, n_term=2, k=15)
        cloned = clone(model)

        assert cloned.n_expansion == 2
        assert cloned.n_term == 2
        assert cloned.k == 15
        # Cloned model should not be fitted
        assert not hasattr(cloned, "is_fitted_") or not cloned.is_fitted_

    def test_get_params(self):
        """get_params() should return all parameters."""
        from pycse.sklearn.sisso import SISSO

        model = SISSO(n_expansion=3, n_term=1, k=10)
        params = model.get_params()

        assert "n_expansion" in params
        assert "n_term" in params
        assert "k" in params
        assert "operators" in params
        assert params["n_expansion"] == 3
        assert params["n_term"] == 1
        assert params["k"] == 10

    def test_set_params(self):
        """set_params() should update parameters."""
        from pycse.sklearn.sisso import SISSO

        model = SISSO(n_expansion=2)
        model.set_params(n_expansion=4, n_term=3)

        assert model.n_expansion == 4
        assert model.n_term == 3

    def test_repr(self):
        """__repr__ should return readable string."""
        from pycse.sklearn.sisso import SISSO

        model = SISSO(n_expansion=2, n_term=1)
        repr_str = repr(model)

        assert "SISSO" in repr_str
        assert "n_expansion=2" in repr_str
        assert "n_term=1" in repr_str


class TestSISSOFeatureNames:
    """Test feature naming functionality."""

    def test_default_feature_names(self, simple_additive_data):
        """Default feature names should be x0, x1, etc."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=1).fit(X, y)

        assert model._feature_names == ["x0", "x1"]

    def test_custom_feature_names(self, simple_additive_data):
        """Custom feature names should be used in equation."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data
        model = SISSO(n_expansion=1, n_term=1, feature_names=["a", "b"]).fit(X, y)

        assert model._feature_names == ["a", "b"]
        # Equation should use custom names
        assert "a" in model.equation_ or "b" in model.equation_


class TestSISSOEquationParsing:
    """Test equation parsing functionality."""

    def test_parse_simple_equation(self):
        """Test parsing simple equations."""
        from pycse.sklearn.sisso import SISSO

        model = SISSO()
        model._feature_names = ["x0", "x1"]

        # Test simple additive
        terms, coeffs, intercept = model._parse_equation("y = 0.5 + 1.0*x0")
        assert "x0" in terms
        assert len(coeffs) == 1
        assert np.isclose(intercept, 0.5)

    def test_parse_equation_with_multiplication(self):
        """Test parsing equations with multiplication."""
        from pycse.sklearn.sisso import SISSO

        model = SISSO()
        model._feature_names = ["x0", "x1"]

        terms, coeffs, intercept = model._parse_equation("y = 2.0*(x0*x1)")
        assert len(terms) >= 1


class TestSISSOEdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature(self):
        """Should work with single feature."""
        from pycse.sklearn.sisso import SISSO

        np.random.seed(42)
        X = np.random.rand(40, 1)
        y = 2 * X[:, 0] + 0.1 * np.random.randn(40)

        model = SISSO(n_expansion=1, n_term=1).fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == (40,)
        assert np.all(np.isfinite(y_pred))

    def test_many_features(self):
        """Should work with many features."""
        from pycse.sklearn.sisso import SISSO

        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = X[:, 0] + X[:, 2] + 0.1 * np.random.randn(50)

        model = SISSO(n_expansion=1, n_term=2, k=10).fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == (50,)

    def test_reproducibility(self, simple_additive_data):
        """Results should be reproducible with same data."""
        from pycse.sklearn.sisso import SISSO

        X, y = simple_additive_data

        model1 = SISSO(n_expansion=1, n_term=1).fit(X, y)
        model2 = SISSO(n_expansion=1, n_term=1).fit(X, y)

        # Equations should be the same
        assert model1.equation_ == model2.equation_


class TestSISSOImport:
    """Test import functionality."""

    def test_lazy_import(self):
        """SISSO should be importable from pycse.sklearn."""
        from pycse.sklearn import SISSO

        assert SISSO is not None

    def test_direct_import(self):
        """SISSO should be importable directly from module."""
        from pycse.sklearn.sisso import SISSO

        assert SISSO is not None
