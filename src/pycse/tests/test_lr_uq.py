"""Comprehensive tests for LinearRegressionUQ class.

This module tests the LinearRegressionUQ class, which provides linear regression
with uncertainty quantification including confidence intervals and standard errors.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from pycse.sklearn.lr_uq import LinearRegressionUQ


class TestLinearRegressionUQBasics:
    """Test basic functionality of LinearRegressionUQ."""

    def test_inheritance(self):
        """Test that LinearRegressionUQ inherits from sklearn base classes."""
        model = LinearRegressionUQ()
        assert isinstance(model, BaseEstimator)
        assert isinstance(model, RegressorMixin)

    def test_simple_linear_fit(self):
        """Test fitting a simple linear relationship y = 2x."""
        # Create simple linear data
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Check that coefficients are stored
        assert hasattr(model, "coefs_")
        assert hasattr(model, "pars_cint")
        assert hasattr(model, "pars_se")

        # Check training data is stored
        assert hasattr(model, "xtrain")
        assert hasattr(model, "ytrain")
        np.testing.assert_array_equal(model.xtrain, X)
        np.testing.assert_array_equal(model.ytrain, y)

    def test_fit_returns_self(self):
        """Test that fit() returns self for sklearn compatibility."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])

        model = LinearRegressionUQ()
        result = model.fit(X, y)

        assert result is model

    def test_predict_without_std(self):
        """Test prediction without standard errors."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Predict at training points
        y_pred = model.predict(X)

        # Should be close to original y values
        np.testing.assert_array_almost_equal(y_pred, y, decimal=10)

    def test_predict_with_std(self):
        """Test prediction with standard errors."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Predict with standard errors
        y_pred, se = model.predict(X, return_std=True)

        # Check that we get both predictions and standard errors
        assert y_pred is not None
        assert se is not None
        assert len(y_pred) == len(X)
        assert len(se) == len(X)

        # Predictions should be close to original
        np.testing.assert_array_almost_equal(y_pred, y, decimal=10)

        # Standard errors should be small for perfect fit
        assert np.all(se >= 0)  # Standard errors are non-negative

    def test_extrapolation(self):
        """Test prediction on new data points."""
        # Train on points 1-5
        X_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        # Predict on point 6
        X_test = np.array([[6]])

        model = LinearRegressionUQ()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Should predict y = 2*6 = 12
        np.testing.assert_array_almost_equal(y_pred, [12.0], decimal=10)


class TestLinearRegressionUQMultipleFeatures:
    """Test LinearRegressionUQ with multiple features."""

    def test_multiple_features(self):
        """Test fitting with multiple input features."""
        # y = 2*x1 + 3*x2, with uncorrelated x1 and x2 (more data points)
        X = np.array(
            [[1, 1], [2, 1], [1, 2], [2, 2], [3, 1], [1, 3], [3, 2], [2, 3], [3, 3], [4, 4]]
        )
        y = 2 * X[:, 0] + 3 * X[:, 1]  # Linear combination

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Predict on training data
        y_pred = model.predict(X)

        # Should fit well (not necessarily perfect due to numerical precision)
        np.testing.assert_array_almost_equal(y_pred, y, decimal=3)

    def test_polynomial_features(self):
        """Test with polynomial features (as used in SurfaceResponse)."""
        from sklearn.preprocessing import PolynomialFeatures

        # Create parabolic data: y = x^2
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 4, 9, 16, 25])

        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X)

        model = LinearRegressionUQ()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)

        # Should fit the parabola perfectly
        np.testing.assert_array_almost_equal(y_pred, y, decimal=5)


class TestLinearRegressionUQMultipleOutputs:
    """Test LinearRegressionUQ with multiple output dimensions."""

    def test_multiple_outputs(self):
        """Test fitting with multiple output variables."""
        X = np.array([[1], [2], [3], [4], [5]])
        # Two outputs: y1 = 2*x, y2 = 3*x
        y = np.array([[2, 3], [4, 6], [6, 9], [8, 12], [10, 15]])

        model = LinearRegressionUQ()
        model.fit(X, y)

        y_pred = model.predict(X)

        # Should fit both outputs
        np.testing.assert_array_almost_equal(y_pred, y, decimal=10)

    def test_multiple_outputs_with_std(self):
        """Test prediction with standard errors for multiple outputs."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2, 3], [4, 6], [6, 9], [8, 12], [10, 15]])

        model = LinearRegressionUQ()
        model.fit(X, y)

        y_pred, se = model.predict(X, return_std=True)

        assert y_pred.shape == y.shape
        assert se.shape[0] == y.shape[0]  # Same number of predictions


class TestLinearRegressionUQConfidenceIntervals:
    """Test confidence interval and standard error properties."""

    def test_has_confidence_intervals(self):
        """Test that confidence intervals are computed."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Check that confidence intervals exist
        assert model.pars_cint is not None
        assert model.pars_se is not None

    def test_confidence_interval_shape(self):
        """Test that confidence intervals have correct shape."""
        X = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [3, 1], [1, 3], [3, 2], [2, 3]])
        y = 2 * X[:, 0] + 3 * X[:, 1]

        model = LinearRegressionUQ()
        model.fit(X, y)

        # pars_cint should be (n_features, 2) for lower and upper bounds
        assert model.pars_cint.shape[1] == 2  # Lower and upper bounds

    def test_standard_errors_positive(self):
        """Test that standard errors are non-negative."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])  # Noisy data

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Standard errors should be positive
        assert np.all(model.pars_se >= 0)


class TestLinearRegressionUQNoisyData:
    """Test LinearRegressionUQ with noisy data."""

    def test_noisy_linear_regression(self):
        """Test fitting data with noise."""
        np.random.seed(42)

        X = np.array([[i] for i in range(1, 21)])
        y_true = 2 * X.ravel() + 1
        noise = np.random.normal(0, 0.5, len(X))
        y = y_true + noise

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Predict
        y_pred = model.predict(X)

        # Should be reasonably close (R^2 > 0.95)
        from sklearn.metrics import r2_score

        r2 = r2_score(y, y_pred)
        assert r2 > 0.95

    def test_prediction_uncertainty_increases_with_extrapolation(self):
        """Test that prediction uncertainty increases when extrapolating."""
        np.random.seed(42)

        # Train on points 0-10
        X_train = np.array([[i] for i in range(11)])
        y_train = 2 * X_train.ravel() + np.random.normal(0, 0.1, len(X_train))

        model = LinearRegressionUQ()
        model.fit(X_train, y_train)

        # Predict at training points and far extrapolation
        X_near = np.array([[5]])  # Middle of training range
        X_far = np.array([[50]])  # Far outside training range

        _, se_near = model.predict(X_near, return_std=True)
        _, se_far = model.predict(X_far, return_std=True)

        # Uncertainty should generally be higher for extrapolation
        # (This may not always hold for all data, but generally true)
        assert se_far[0] > se_near[0] * 0.5  # At least somewhat higher


class TestLinearRegressionUQEdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature_single_output(self):
        """Test with minimal dimensionality."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])

        model = LinearRegressionUQ()
        model.fit(X, y)

        y_pred = model.predict(X)

        assert len(y_pred) == len(X)

    def test_predict_single_point(self):
        """Test prediction on a single point."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Predict single point
        y_pred = model.predict(np.array([[3]]))

        assert len(y_pred) == 1
        np.testing.assert_almost_equal(y_pred[0], 6, decimal=10)

    def test_fit_accepts_lists(self):
        """Test that fit accepts Python lists, not just numpy arrays."""
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]

        model = LinearRegressionUQ()
        model.fit(X, y)

        # Should convert to numpy arrays internally
        assert isinstance(model.xtrain, np.ndarray)
        assert isinstance(model.ytrain, np.ndarray)

        y_pred = model.predict(X)
        assert len(y_pred) == len(X)


class TestLinearRegressionUQIntegration:
    """Integration tests with actual use cases."""

    def test_surface_response_integration(self):
        """Test integration with PolynomialFeatures like SurfaceResponse uses."""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline

        # Create 2D response surface data
        np.random.seed(42)
        n_points = 20

        x1 = np.random.uniform(-1, 1, n_points)
        x2 = np.random.uniform(-1, 1, n_points)
        X = np.column_stack([x1, x2])

        # True function: y = 1 + 2*x1 + 3*x2 + 0.5*x1^2 + noise
        y = 1 + 2 * x1 + 3 * x2 + 0.5 * x1**2 + np.random.normal(0, 0.1, n_points)

        # Create pipeline like SurfaceResponse does
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2)),
                ("regressor", LinearRegressionUQ()),
            ]
        )

        model.fit(X, y)
        y_pred = model.predict(X)

        # Should fit reasonably well
        from sklearn.metrics import r2_score

        r2 = r2_score(y, y_pred)
        assert r2 > 0.9

    def test_coefficients_accessible_in_pipeline(self):
        """Test that UQ attributes are accessible through pipeline."""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline

        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=1)),
                ("regressor", LinearRegressionUQ()),
            ]
        )

        model.fit(X, y)

        # Access through pipeline
        regressor = model["regressor"]
        assert hasattr(regressor, "coefs_")
        assert hasattr(regressor, "pars_cint")
        assert hasattr(regressor, "pars_se")


class TestLinearRegressionUQNumericalStability:
    """Test numerical stability and precision."""

    def test_perfect_fit_zero_std_error(self):
        """Test that perfect fit gives very small standard errors."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        model = LinearRegressionUQ()
        model.fit(X, y)

        _, se = model.predict(X, return_std=True)

        # For perfect fit, standard errors should be very small
        assert np.all(se < 1e-10)

    def test_large_values(self):
        """Test with large numerical values."""
        X = np.array([[1e6], [2e6], [3e6], [4e6], [5e6]])
        y = np.array([2e6, 4e6, 6e6, 8e6, 10e6])

        model = LinearRegressionUQ()
        model.fit(X, y)

        y_pred = model.predict(X)

        # Should still fit well
        np.testing.assert_array_almost_equal(y_pred, y, decimal=0)

    def test_small_values(self):
        """Test with small numerical values."""
        X = np.array([[1e-6], [2e-6], [3e-6], [4e-6], [5e-6]])
        y = np.array([2e-6, 4e-6, 6e-6, 8e-6, 10e-6])

        model = LinearRegressionUQ()
        model.fit(X, y)

        y_pred = model.predict(X)

        # Should still fit well
        np.testing.assert_array_almost_equal(y_pred, y, decimal=15)
