"""Tests for JAXPeriodicRegressor - Periodic Neural Network with LLPR."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pycse.sklearn.jax_periodic import JAXPeriodicRegressor


# Use small networks and few epochs for fast tests
_TEST_EPOCHS = 10
_TEST_HIDDEN_DIMS = (8, 8)


@pytest.fixture
def periodic_sine_data():
    """Generate data with sine periodicity."""
    np.random.seed(42)
    X = np.random.uniform(0, 4 * np.pi, (80, 1))
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(80)
    return X, y


@pytest.fixture
def periodic_2d_data():
    """Generate data with periodicity in first dimension."""
    np.random.seed(42)
    X = np.random.uniform(0, 2 * np.pi, (80, 2))
    # Periodic in x0, linear in x1
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * np.random.randn(80)
    return X, y


@pytest.fixture
def multi_periodic_data():
    """Generate data with multiple periodic features."""
    np.random.seed(42)
    X = np.random.uniform(0, 2 * np.pi, (80, 3))
    # Periodic in x0 and x2, linear in x1
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1] + np.cos(X[:, 2]) + 0.1 * np.random.randn(80)
    return X, y


@pytest.fixture
def simple_linear_data():
    """Generate simple linear data for basic tests."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = X[:, 0] + 2 * X[:, 1] + 0.1 * np.random.randn(50)
    return X, y


class TestJAXPeriodicInitialization:
    """Test initialization of JAXPeriodicRegressor."""

    def test_default_initialization(self):
        """Test default parameter values."""
        model = JAXPeriodicRegressor()

        assert model.hidden_dims == (32, 32)
        assert model.periodicity is None
        assert model.n_harmonics == 5
        assert model.activation == "silu"
        assert model.learning_rate == 5e-3
        assert model.weight_decay == 0.0
        assert model.epochs == 500
        assert model.batch_size == 32
        assert model.standardize_X is True
        assert model.standardize_y is True
        assert model.alpha_squared == "auto"
        assert model.zeta_squared == "auto"
        assert model.val_size == 0.1
        assert model.random_state == 42
        assert model.verbose is False

    def test_custom_initialization(self):
        """Test custom parameter values."""
        model = JAXPeriodicRegressor(
            hidden_dims=(64, 32, 16),
            periodicity={0: 2 * np.pi, 2: 1.0},
            n_harmonics=10,
            activation="relu",
            learning_rate=5e-4,
            weight_decay=1e-4,
            epochs=1000,
            batch_size=64,
            standardize_X=False,
            standardize_y=False,
            alpha_squared=1.0,
            zeta_squared=1e-6,
            val_size=0.2,
            random_state=123,
            verbose=True,
        )

        assert model.hidden_dims == (64, 32, 16)
        assert model.periodicity == {0: 2 * np.pi, 2: 1.0}
        assert model.n_harmonics == 10
        assert model.activation == "relu"
        assert model.learning_rate == 5e-4
        assert model.weight_decay == 1e-4
        assert model.epochs == 1000
        assert model.batch_size == 64
        assert model.standardize_X is False
        assert model.standardize_y is False
        assert model.alpha_squared == 1.0
        assert model.zeta_squared == 1e-6
        assert model.val_size == 0.2
        assert model.random_state == 123
        assert model.verbose is True


class TestJAXPeriodicBasicFunctionality:
    """Test basic fit/predict functionality."""

    def test_fit_returns_self(self, simple_linear_data):
        """Test that fit returns self."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        result = model.fit(X, y)
        assert result is model

    def test_fit_predict_basic(self, simple_linear_data):
        """Test basic fit and predict cycle."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_fit_predict_with_periodicity(self, periodic_2d_data):
        """Test fit/predict with periodic features."""
        X, y = periodic_2d_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            n_harmonics=3,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_predict_shape_single_sample(self, simple_linear_data):
        """Test prediction on single sample."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred = model.predict(X[0:1])
        assert y_pred.shape == (1,)

    def test_predict_shape_batch(self, simple_linear_data):
        """Test prediction on batch."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred = model.predict(X[:20])
        assert y_pred.shape == (20,)

    def test_fit_with_1d_y(self, simple_linear_data):
        """Test fit works with 1D y array."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y.ravel())
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)

    def test_fit_with_2d_y(self, simple_linear_data):
        """Test fit works with 2D y array."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y.reshape(-1, 1))
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)


class TestJAXPeriodicPeriodicity:
    """Test periodicity guarantees."""

    def test_periodicity_exact(self, periodic_sine_data):
        """Test that predictions are periodic with specified period."""
        X, y = periodic_sine_data

        model = JAXPeriodicRegressor(
            periodicity={0: 2 * np.pi},
            n_harmonics=5,
            epochs=50,  # Need more epochs for this test
            hidden_dims=_TEST_HIDDEN_DIMS,
            random_state=42,
        )
        model.fit(X, y)

        # Test points
        x_base = np.array([[0.5], [1.0], [2.0]])
        x_shifted = x_base + 2 * np.pi

        y_base = model.predict(x_base)
        y_shifted = model.predict(x_shifted)

        # Predictions should be identical (within numerical precision)
        np.testing.assert_allclose(y_base, y_shifted, rtol=1e-10)

    def test_periodicity_different_period(self):
        """Test periodicity with a different period."""
        np.random.seed(42)
        period = 1.0
        X = np.random.uniform(0, 3 * period, (60, 1))
        y = np.sin(2 * np.pi * X[:, 0] / period)

        model = JAXPeriodicRegressor(
            periodicity={0: period},
            n_harmonics=3,
            epochs=30,
            hidden_dims=_TEST_HIDDEN_DIMS,
        )
        model.fit(X, y)

        x_base = np.array([[0.3]])
        x_shifted = x_base + period

        y_base = model.predict(x_base)
        y_shifted = model.predict(x_shifted)

        np.testing.assert_allclose(y_base, y_shifted, rtol=1e-10)

    def test_mixed_periodic_nonperiodic(self, periodic_2d_data):
        """Test with mix of periodic and non-periodic features."""
        X, y = periodic_2d_data

        model = JAXPeriodicRegressor(
            periodicity={0: 2 * np.pi},  # Only x0 is periodic
            n_harmonics=3,
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
        )
        model.fit(X, y)

        # Periodicity in x0
        x1 = np.array([[0.5, 1.0]])
        x2 = np.array([[0.5 + 2 * np.pi, 1.0]])
        np.testing.assert_allclose(model.predict(x1), model.predict(x2), rtol=1e-10)

        # No periodicity in x1 (predictions should differ)
        x3 = np.array([[0.5, 2.0]])
        assert not np.allclose(model.predict(x1), model.predict(x3))

    def test_multiple_periodic_features(self, multi_periodic_data):
        """Test with multiple periodic features."""
        X, y = multi_periodic_data

        model = JAXPeriodicRegressor(
            periodicity={0: 2 * np.pi, 2: 2 * np.pi},
            n_harmonics=3,
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
        )
        model.fit(X, y)

        # Both x0 and x2 are periodic
        x_base = np.array([[0.5, 1.0, 0.8]])
        x_shift_0 = np.array([[0.5 + 2 * np.pi, 1.0, 0.8]])
        x_shift_2 = np.array([[0.5, 1.0, 0.8 + 2 * np.pi]])

        y_base = model.predict(x_base)
        np.testing.assert_allclose(y_base, model.predict(x_shift_0), rtol=1e-10)
        np.testing.assert_allclose(y_base, model.predict(x_shift_2), rtol=1e-10)


class TestJAXPeriodicUncertainty:
    """Test LLPR uncertainty quantification."""

    def test_predict_with_uncertainty_shapes(self, simple_linear_data):
        """Test shapes of uncertainty predictions."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred, y_std = model.predict_with_uncertainty(X)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)
        assert np.all(y_std >= 0), "Standard deviations should be non-negative"

    def test_predict_with_uncertainty_variance(self, simple_linear_data):
        """Test variance output option."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred, y_var = model.predict_with_uncertainty(X, return_std=False)
        y_pred2, y_std = model.predict_with_uncertainty(X, return_std=True)

        assert np.allclose(y_pred, y_pred2)
        assert np.allclose(y_var, y_std**2, rtol=1e-5)

    def test_uncertainty_increases_away_from_training(self, simple_linear_data):
        """Test that uncertainty generally increases away from training data."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        # Points near training data
        X_near = X[:10]
        # Points far from training data
        X_far = X[:10] + 10.0

        _, std_near = model.predict_with_uncertainty(X_near)
        _, std_far = model.predict_with_uncertainty(X_far)

        # Uncertainty should generally be higher far from training data
        assert np.mean(std_far) > np.mean(std_near)

    def test_calibration_parameters_set(self, simple_linear_data):
        """Test that LLPR calibration parameters are set after fit."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            alpha_squared="auto",
            zeta_squared="auto",
        )
        model.fit(X, y)

        assert hasattr(model, "alpha_squared_")
        assert hasattr(model, "zeta_squared_")
        assert model.alpha_squared_ > 0
        assert model.zeta_squared_ > 0

    def test_manual_calibration_parameters(self, simple_linear_data):
        """Test with manually specified calibration parameters."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, alpha_squared=2.0, zeta_squared=1e-4
        )
        model.fit(X, y)

        assert model.alpha_squared_ == 2.0
        assert model.zeta_squared_ == 1e-4


class TestJAXPeriodicFeatureExpansion:
    """Test Fourier feature expansion."""

    def test_n_expanded_features_no_periodic(self, simple_linear_data):
        """Test expanded features count with no periodicity."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, periodicity=None
        )
        model.fit(X, y)

        # No expansion: same as input
        assert model.n_expanded_features_ == X.shape[1]

    def test_n_expanded_features_single_periodic(self):
        """Test expanded features count with single periodic feature."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = X[:, 0] + X[:, 1] + X[:, 2]

        n_harmonics = 4
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={1: 1.0},
            n_harmonics=n_harmonics,
        )
        model.fit(X, y)

        # 2 non-periodic + 1 periodic * 2 * n_harmonics
        expected = 2 + 2 * n_harmonics
        assert model.n_expanded_features_ == expected

    def test_n_expanded_features_multiple_periodic(self):
        """Test expanded features count with multiple periodic features."""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        y = np.sum(X, axis=1)

        n_harmonics = 3
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 1.0, 2: 2.0},
            n_harmonics=n_harmonics,
        )
        model.fit(X, y)

        # 2 non-periodic + 2 periodic * 2 * n_harmonics
        expected = 2 + 2 * 2 * n_harmonics
        assert model.n_expanded_features_ == expected

    def test_get_fourier_features(self, simple_linear_data):
        """Test get_fourier_features method."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            n_harmonics=2,
        )
        model.fit(X, y)

        features = model.get_fourier_features(X[:5])
        assert features.shape == (5, model.n_expanded_features_)


class TestJAXPeriodicScoring:
    """Test scoring functionality."""

    def test_score_reasonable(self, periodic_sine_data):
        """Test that R² score is reasonable for periodic data."""
        X, y = periodic_sine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = JAXPeriodicRegressor(
            epochs=50,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            n_harmonics=3,
            random_state=42,
        )
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        # For periodic data with matching period, score should be good
        assert score > 0.5, f"Score {score} is too low for periodic data"

    def test_score_matches_sklearn_convention(self, simple_linear_data):
        """Test that score follows sklearn R² convention."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred = model.predict(X)
        score = model.score(X, y)

        # Manual R² calculation
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        expected_score = 1 - ss_res / ss_tot

        assert np.isclose(score, expected_score)


class TestJAXPeriodicSklearnCompatibility:
    """Test sklearn compatibility features."""

    def test_get_params(self):
        """Test get_params method."""
        model = JAXPeriodicRegressor(
            hidden_dims=(64, 64), periodicity={0: 1.0}, n_harmonics=4, epochs=100
        )
        params = model.get_params()

        assert params["hidden_dims"] == (64, 64)
        assert params["periodicity"] == {0: 1.0}
        assert params["n_harmonics"] == 4
        assert params["epochs"] == 100

    def test_set_params(self):
        """Test set_params method."""
        model = JAXPeriodicRegressor()
        model.set_params(epochs=200, n_harmonics=8)

        assert model.epochs == 200
        assert model.n_harmonics == 8

    def test_clone(self, simple_linear_data):
        """Test that model can be cloned."""
        from sklearn.base import clone

        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, periodicity={0: 1.0}
        )
        model.fit(X, y)

        cloned = clone(model)
        assert cloned.epochs == _TEST_EPOCHS
        assert cloned.periodicity == {0: 1.0}
        assert not hasattr(cloned, "params_")

    def test_pipeline_compatibility(self, simple_linear_data):
        """Test usage in sklearn Pipeline."""
        X, y = simple_linear_data

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    JAXPeriodicRegressor(
                        epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, standardize_X=False
                    ),
                ),
            ]
        )

        pipe.fit(X, y)
        y_pred = pipe.predict(X)

        assert y_pred.shape == (len(X),)

    def test_gridsearchcv_compatibility(self, simple_linear_data):
        """Test usage with GridSearchCV."""
        X, y = simple_linear_data

        model = JAXPeriodicRegressor(epochs=5, hidden_dims=(4,))
        param_grid = {"hidden_dims": [(4,), (4, 4)], "n_harmonics": [2, 3]}

        # Just test that it runs without error
        grid = GridSearchCV(model, param_grid, cv=2, scoring="r2")
        grid.fit(X, y)

        assert hasattr(grid, "best_params_")


class TestJAXPeriodicStandardization:
    """Test standardization options."""

    def test_no_standardization(self, simple_linear_data):
        """Test with no standardization."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            standardize_X=False,
            standardize_y=False,
        )
        model.fit(X, y)

        assert model.scaler_X_ is None
        assert model.scaler_y_ is None

        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)

    def test_x_standardization_only(self, simple_linear_data):
        """Test with only X standardization."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            standardize_X=True,
            standardize_y=False,
        )
        model.fit(X, y)

        # Scaler is only created if there are non-periodic features
        assert model.scaler_X_ is not None
        assert model.scaler_y_ is None

    def test_y_standardization_only(self, simple_linear_data):
        """Test with only y standardization."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            standardize_X=False,
            standardize_y=True,
        )
        model.fit(X, y)

        assert model.scaler_X_ is None
        assert model.scaler_y_ is not None

    def test_periodic_features_not_standardized(self):
        """Test that periodic features are not standardized."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (50, 2))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},  # x0 is periodic, x1 is not
            standardize_X=True,
        )
        model.fit(X, y)

        # Only non-periodic feature should be in scaler
        # scaler_X_ only applies to non-periodic features
        assert model.scaler_X_ is not None


class TestJAXPeriodicReproducibility:
    """Test reproducibility with random state."""

    def test_reproducibility(self, simple_linear_data):
        """Test that same random_state gives same results."""
        X, y = simple_linear_data

        model1 = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, random_state=42
        )
        model1.fit(X, y)
        y_pred1 = model1.predict(X)

        model2 = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, random_state=42
        )
        model2.fit(X, y)
        y_pred2 = model2.predict(X)

        np.testing.assert_allclose(y_pred1, y_pred2, rtol=1e-5)

    def test_different_random_states(self, simple_linear_data):
        """Test that different random_states give different results."""
        X, y = simple_linear_data

        model1 = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, random_state=42
        )
        model1.fit(X, y)
        y_pred1 = model1.predict(X)

        model2 = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, random_state=123
        )
        model2.fit(X, y)
        y_pred2 = model2.predict(X)

        # Predictions should be different (not exactly equal)
        assert not np.allclose(y_pred1, y_pred2)


class TestJAXPeriodicActivations:
    """Test different activation functions."""

    def test_silu_activation(self, simple_linear_data):
        """Test with silu activation."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, activation="silu"
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_softplus_activation(self, simple_linear_data):
        """Test with softplus activation."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, activation="softplus"
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_relu_activation(self, simple_linear_data):
        """Test with relu activation."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, activation="relu"
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_tanh_activation(self, simple_linear_data):
        """Test with tanh activation."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, activation="tanh"
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestJAXPeriodicEdgeCases:
    """Test edge cases."""

    def test_single_feature_periodic(self):
        """Test with single periodic feature."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (30, 1))
        y = np.sin(X[:, 0])

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            n_harmonics=3,
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (30,)

    def test_all_features_periodic(self):
        """Test with all features periodic."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (30, 3))
        y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.sin(X[:, 2])

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi, 1: 2 * np.pi, 2: 2 * np.pi},
            n_harmonics=2,
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (30,)

    def test_small_dataset(self):
        """Test with small dataset."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = X[:, 0] + X[:, 1]

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, val_size=0.0
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (20,)

    def test_single_harmonic(self):
        """Test with single harmonic."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (30, 1))
        y = np.sin(X[:, 0])

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            n_harmonics=1,  # Single harmonic
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (30,)
        assert model.n_expanded_features_ == 2  # sin + cos only

    def test_many_harmonics(self):
        """Test with many harmonics."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (30, 1))
        y = np.sin(X[:, 0]) + 0.5 * np.sin(2 * X[:, 0]) + 0.25 * np.sin(3 * X[:, 0])

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            n_harmonics=10,
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (30,)
        assert model.n_expanded_features_ == 20  # 10 * 2 (sin + cos)


class TestJAXPeriodicAttributes:
    """Test fitted attributes."""

    def test_n_features_in(self, simple_linear_data):
        """Test n_features_in_ attribute."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        assert model.n_features_in_ == X.shape[1]

    def test_loss_history(self, simple_linear_data):
        """Test loss_history_ attribute."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        assert hasattr(model, "loss_history_")
        assert len(model.loss_history_) == _TEST_EPOCHS

    def test_params_stored(self, simple_linear_data):
        """Test that params_ is stored after fit."""
        X, y = simple_linear_data
        model = JAXPeriodicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        assert hasattr(model, "params_")
        assert "W" in model.params_
        assert "b" in model.params_

    def test_periodicity_attributes(self):
        """Test periodicity-related attributes."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.sum(X, axis=1)

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 1.0, 2: 2.0},
            n_harmonics=3,
        )
        model.fit(X, y)

        assert hasattr(model, "periodicity_")
        assert model.periodicity_ == {0: 1.0, 2: 2.0}
        assert hasattr(model, "periodic_indices_")
        assert hasattr(model, "periods_")
        assert hasattr(model, "n_expanded_features_")


class TestJAXPeriodicValidation:
    """Test input validation."""

    def test_invalid_periodicity_index(self):
        """Test that invalid periodicity index raises error."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        model = JAXPeriodicRegressor(periodicity={5: 1.0})  # Index 5 out of range

        with pytest.raises(ValueError, match="out of range"):
            model.fit(X, y)

    def test_negative_period(self):
        """Test that negative period raises error."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        model = JAXPeriodicRegressor(periodicity={0: -1.0})  # Negative period

        with pytest.raises(ValueError, match="Period must be positive"):
            model.fit(X, y)

    def test_zero_period(self):
        """Test that zero period raises error."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        model = JAXPeriodicRegressor(periodicity={0: 0.0})  # Zero period

        with pytest.raises(ValueError, match="Period must be positive"):
            model.fit(X, y)


class TestJAXPeriodicLearnablePeriods:
    """Test learnable period functionality."""

    def test_learn_period_default_false(self):
        """Test that learn_period defaults to False."""
        model = JAXPeriodicRegressor()
        assert model.learn_period is False

    def test_learn_period_initialization(self):
        """Test initialization with learn_period=True."""
        model = JAXPeriodicRegressor(
            periodicity={0: 2 * np.pi},
            learn_period=True,
            period_reg=0.05,
        )
        assert model.learn_period is True
        assert model.period_reg == 0.05

    def test_learn_period_params_contain_period_raw(self):
        """Test that params contain period_raw when learn_period=True."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (50, 1))
        y = np.sin(X[:, 0])

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            learn_period=True,
        )
        model.fit(X, y)

        assert "period_raw" in model.params_

    def test_learn_period_no_period_raw_when_false(self):
        """Test that params don't contain period_raw when learn_period=False."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (50, 1))
        y = np.sin(X[:, 0])

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            learn_period=False,
        )
        model.fit(X, y)

        assert "period_raw" not in model.params_

    def test_learned_periods_attribute(self):
        """Test that learned_periods_ is set correctly."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (50, 1))
        y = np.sin(X[:, 0])

        # With learn_period=True
        model_learn = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            learn_period=True,
        )
        model_learn.fit(X, y)
        assert model_learn.learned_periods_ is not None
        assert 0 in model_learn.learned_periods_
        assert model_learn.learned_periods_[0] > 0

        # With learn_period=False
        model_fixed = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            learn_period=False,
        )
        model_fixed.fit(X, y)
        assert model_fixed.learned_periods_ is None

    def test_learn_period_discovers_correct_period(self):
        """Test that learnable periods can discover the correct period."""
        np.random.seed(42)
        true_period = 5.0
        n_samples = 150

        X = np.random.uniform(0, 3 * true_period, (n_samples, 1))
        y = np.sin(2 * np.pi * X[:, 0] / true_period) + 0.05 * np.random.randn(n_samples)

        # Start with wrong initial guess
        model = JAXPeriodicRegressor(
            epochs=100,  # More epochs for learning
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 6.0},  # Initial guess (wrong)
            learn_period=True,
            period_reg=0.001,  # Low regularization to allow learning
            n_harmonics=3,
            random_state=42,
        )
        model.fit(X, y)

        # Check that period moved toward true value
        learned = model.learned_periods_[0]
        initial = 6.0

        # Learned period should be closer to true period than initial guess
        assert abs(learned - true_period) < abs(initial - true_period)

    def test_learn_period_regularization_effect(self):
        """Test that period_reg controls how much period can change."""
        np.random.seed(42)
        true_period = 5.0
        initial_guess = 7.0

        X = np.random.uniform(0, 3 * true_period, (100, 1))
        y = np.sin(2 * np.pi * X[:, 0] / true_period) + 0.05 * np.random.randn(100)

        # High regularization: period should stay close to initial
        model_high_reg = JAXPeriodicRegressor(
            epochs=50,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: initial_guess},
            learn_period=True,
            period_reg=10.0,  # High regularization
            n_harmonics=3,
            random_state=42,
        )
        model_high_reg.fit(X, y)

        # Low regularization: period should move more
        model_low_reg = JAXPeriodicRegressor(
            epochs=50,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: initial_guess},
            learn_period=True,
            period_reg=0.001,  # Low regularization
            n_harmonics=3,
            random_state=42,
        )
        model_low_reg.fit(X, y)

        # High reg should stay closer to initial
        high_reg_change = abs(model_high_reg.learned_periods_[0] - initial_guess)
        low_reg_change = abs(model_low_reg.learned_periods_[0] - initial_guess)

        assert high_reg_change < low_reg_change

    def test_learn_period_with_multiple_periodic_features(self):
        """Test learnable periods with multiple periodic features."""
        np.random.seed(42)
        n_samples = 100
        period1, period2 = 4.0, 6.0

        X = np.column_stack(
            [
                np.random.uniform(0, 2 * period1, n_samples),
                np.random.uniform(0, 2 * period2, n_samples),
            ]
        )
        y = (
            np.sin(2 * np.pi * X[:, 0] / period1)
            + np.cos(2 * np.pi * X[:, 1] / period2)
            + 0.05 * np.random.randn(n_samples)
        )

        model = JAXPeriodicRegressor(
            epochs=50,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 5.0, 1: 7.0},  # Initial guesses
            learn_period=True,
            period_reg=0.01,
            n_harmonics=3,
            random_state=42,
        )
        model.fit(X, y)

        # Both features should have learned periods
        assert 0 in model.learned_periods_
        assert 1 in model.learned_periods_
        assert model.learned_periods_[0] > 0
        assert model.learned_periods_[1] > 0

    def test_learn_period_sklearn_compatibility(self):
        """Test that learn_period works with sklearn utilities."""
        from sklearn.base import clone

        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (50, 1))
        y = np.sin(X[:, 0])

        model = JAXPeriodicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            periodicity={0: 2 * np.pi},
            learn_period=True,
            period_reg=0.1,
        )
        model.fit(X, y)

        # Clone should preserve learn_period setting
        cloned = clone(model)
        assert cloned.learn_period is True
        assert cloned.period_reg == 0.1
        assert not hasattr(cloned, "learned_periods_")

    def test_learn_period_get_set_params(self):
        """Test get_params and set_params with learn_period."""
        model = JAXPeriodicRegressor(
            periodicity={0: 2 * np.pi},
            learn_period=True,
            period_reg=0.5,
        )

        params = model.get_params()
        assert params["learn_period"] is True
        assert params["period_reg"] == 0.5

        model.set_params(learn_period=False, period_reg=0.2)
        assert model.learn_period is False
        assert model.period_reg == 0.2
