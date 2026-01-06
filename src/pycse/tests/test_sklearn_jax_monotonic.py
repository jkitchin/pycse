"""Tests for JAXMonotonicRegressor - Monotonic Neural Network with LLPR."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pycse.sklearn.jax_monotonic import JAXMonotonicRegressor


# Use small networks and few epochs for fast tests
_TEST_EPOCHS = 10
_TEST_HIDDEN_DIMS = (8, 8)

# Mark all tests in this module as slow (ML model training)
pytestmark = pytest.mark.slow


@pytest.fixture
def monotonic_increasing_data():
    """Generate data that is monotonically increasing in all features."""
    np.random.seed(42)
    X = np.random.randn(60, 2) * 2
    # y = 2*x1 + 3*x2 (linear, increasing in both)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * np.random.randn(60)
    return X, y


@pytest.fixture
def monotonic_decreasing_data():
    """Generate data that is monotonically decreasing in all features."""
    np.random.seed(42)
    X = np.random.randn(60, 2) * 2
    # y = -2*x1 - 3*x2 (linear, decreasing in both)
    y = -2 * X[:, 0] - 3 * X[:, 1] + 0.1 * np.random.randn(60)
    return X, y


@pytest.fixture
def mixed_monotonicity_data():
    """Generate data with mixed monotonicity."""
    np.random.seed(42)
    X = np.random.randn(60, 3) * 2
    # y = 2*x1 - x2 + sin(x3) (increasing in x1, decreasing in x2, nonmonotonic in x3)
    y = 2 * X[:, 0] - X[:, 1] + np.sin(X[:, 2]) + 0.1 * np.random.randn(60)
    return X, y


@pytest.fixture
def simple_linear_data():
    """Generate simple linear data for basic tests."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = X[:, 0] + 2 * X[:, 1] + 0.1 * np.random.randn(50)
    return X, y


class TestJAXMonotonicInitialization:
    """Test initialization of JAXMonotonicRegressor."""

    def test_default_initialization(self):
        """Test default parameter values."""
        model = JAXMonotonicRegressor()

        assert model.hidden_dims == (32, 32)
        assert model.monotonicity == 1
        assert model.activation == "softplus"
        assert model.nonneg_param == "softplus"
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
        model = JAXMonotonicRegressor(
            hidden_dims=(64, 32, 16),
            monotonicity=-1,
            activation="relu",
            nonneg_param="square",
            learning_rate=5e-4,
            weight_decay=1e-4,
            epochs=100,
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
        assert model.monotonicity == -1
        assert model.activation == "relu"
        assert model.nonneg_param == "square"
        assert model.learning_rate == 5e-4
        assert model.weight_decay == 1e-4
        assert model.epochs == 100
        assert model.batch_size == 64
        assert model.standardize_X is False
        assert model.standardize_y is False
        assert model.alpha_squared == 1.0
        assert model.zeta_squared == 1e-6
        assert model.val_size == 0.2
        assert model.random_state == 123
        assert model.verbose is True

    def test_list_monotonicity(self):
        """Test initialization with list monotonicity."""
        model = JAXMonotonicRegressor(monotonicity=[1, -1, 0])
        assert model.monotonicity == [1, -1, 0]


class TestJAXMonotonicBasicFunctionality:
    """Test basic fit/predict functionality."""

    def test_fit_returns_self(self, simple_linear_data):
        """Test that fit returns self."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        result = model.fit(X, y)
        assert result is model

    def test_fit_predict_basic(self, simple_linear_data):
        """Test basic fit and predict cycle."""
        X, y = simple_linear_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_predict_shape_single_sample(self, simple_linear_data):
        """Test prediction on single sample."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred = model.predict(X[0:1])
        assert y_pred.shape == (1,)

    def test_predict_shape_batch(self, simple_linear_data):
        """Test prediction on batch."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred = model.predict(X[:20])
        assert y_pred.shape == (20,)

    def test_fit_with_1d_y(self, simple_linear_data):
        """Test fit works with 1D y array."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y.ravel())
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)

    def test_fit_with_2d_y(self, simple_linear_data):
        """Test fit works with 2D y array."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y.reshape(-1, 1))
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)


class TestJAXMonotonicMonotonicity:
    """Test monotonicity guarantees."""

    def test_monotonic_increasing_gradient(self, monotonic_increasing_data):
        """Test that gradients are nonnegative for increasing monotonicity."""
        X, y = monotonic_increasing_data

        model = JAXMonotonicRegressor(
            monotonicity=1,  # All increasing
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            random_state=42,
        )
        model.fit(X, y)

        grads = model.predict_gradient(X)

        # All gradients should be >= 0 for increasing monotonicity
        # Use small tolerance for numerical precision
        assert np.all(grads >= -1e-6), f"Found negative gradients: {grads.min()}"

    def test_monotonic_decreasing_gradient(self, monotonic_decreasing_data):
        """Test that gradients are nonpositive for decreasing monotonicity."""
        X, y = monotonic_decreasing_data

        model = JAXMonotonicRegressor(
            monotonicity=-1,  # All decreasing
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            random_state=42,
        )
        model.fit(X, y)

        grads = model.predict_gradient(X)

        # All gradients should be <= 0 for decreasing monotonicity
        assert np.all(grads <= 1e-6), f"Found positive gradients: {grads.max()}"

    def test_mixed_monotonicity_gradient(self, mixed_monotonicity_data):
        """Test gradients with mixed monotonicity constraints."""
        X, y = mixed_monotonicity_data

        model = JAXMonotonicRegressor(
            monotonicity=[1, -1, 0],  # x0 inc, x1 dec, x2 unconstrained
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            random_state=42,
        )
        model.fit(X, y)

        grads = model.predict_gradient(X)

        # x0: increasing, gradient >= 0
        assert np.all(grads[:, 0] >= -1e-6), "x0 has negative gradients"
        # x1: decreasing, gradient <= 0
        assert np.all(grads[:, 1] <= 1e-6), "x1 has positive gradients"
        # x2: unconstrained, can have any sign (no assertion needed)

    def test_monotonicity_on_line(self, simple_linear_data):
        """Test monotonicity along a line in input space."""
        X, y = simple_linear_data

        model = JAXMonotonicRegressor(
            monotonicity=1,
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            random_state=42,
        )
        model.fit(X, y)

        # Create points along a line
        t = np.linspace(-2, 2, 50)
        X_line = np.column_stack([t, t])  # Diagonal line

        y_pred = model.predict(X_line)

        # Predictions should be monotonically increasing along the line
        # (since both x1 and x2 increase and have positive monotonicity)
        diffs = np.diff(y_pred)
        assert np.all(diffs >= -1e-6), "Not monotonic along the line"


class TestJAXMonotonicUncertainty:
    """Test LLPR uncertainty quantification."""

    def test_predict_with_uncertainty_shapes(self, simple_linear_data):
        """Test shapes of uncertainty predictions."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred, y_std = model.predict_with_uncertainty(X)

        assert y_pred.shape == (len(X),)
        assert y_std.shape == (len(X),)
        assert np.all(y_std >= 0), "Standard deviations should be non-negative"

    def test_predict_with_uncertainty_variance(self, simple_linear_data):
        """Test variance output option."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred, y_var = model.predict_with_uncertainty(X, return_std=False)
        y_pred2, y_std = model.predict_with_uncertainty(X, return_std=True)

        assert np.allclose(y_pred, y_pred2)
        assert np.allclose(y_var, y_std**2, rtol=1e-5)

    def test_uncertainty_increases_away_from_training(self, simple_linear_data):
        """Test that uncertainty generally increases away from training data."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
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
        model = JAXMonotonicRegressor(
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
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, alpha_squared=2.0, zeta_squared=1e-4
        )
        model.fit(X, y)

        assert model.alpha_squared_ == 2.0
        assert model.zeta_squared_ == 1e-4


class TestJAXMonotonicGradients:
    """Test gradient computation."""

    def test_gradient_shape(self, simple_linear_data):
        """Test gradient output shape."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        grads = model.predict_gradient(X)

        assert grads.shape == X.shape

    def test_gradient_single_sample(self, simple_linear_data):
        """Test gradient on single sample."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        grads = model.predict_gradient(X[0:1])

        assert grads.shape == (1, X.shape[1])

    def test_gradient_finite(self, simple_linear_data):
        """Test that gradients are finite."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        grads = model.predict_gradient(X)

        assert np.all(np.isfinite(grads))


class TestJAXMonotonicScoring:
    """Test scoring functionality."""

    def test_score_reasonable(self, monotonic_increasing_data):
        """Test that R² score is reasonable for monotonic data."""
        X, y = monotonic_increasing_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = JAXMonotonicRegressor(epochs=10, hidden_dims=_TEST_HIDDEN_DIMS, random_state=42)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        # For nearly linear monotonic data, score should be reasonable
        assert score > 0.0, f"Score {score} is too low for monotonic data"

    def test_score_matches_sklearn_convention(self, simple_linear_data):
        """Test that score follows sklearn R² convention."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        y_pred = model.predict(X)
        score = model.score(X, y)

        # Manual R² calculation
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        expected_score = 1 - ss_res / ss_tot

        assert np.isclose(score, expected_score)


class TestJAXMonotonicSklearnCompatibility:
    """Test sklearn compatibility features."""

    def test_get_params(self):
        """Test get_params method."""
        model = JAXMonotonicRegressor(hidden_dims=(64, 64), monotonicity=-1, epochs=10)
        params = model.get_params()

        assert params["hidden_dims"] == (64, 64)
        assert params["monotonicity"] == -1
        assert params["epochs"] == 10

    def test_set_params(self):
        """Test set_params method."""
        model = JAXMonotonicRegressor()
        model.set_params(epochs=100, monotonicity=-1)

        assert model.epochs == 100
        assert model.monotonicity == -1

    def test_clone(self, simple_linear_data):
        """Test that model can be cloned."""
        from sklearn.base import clone

        X, y = simple_linear_data
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, monotonicity=[1, 1]
        )
        model.fit(X, y)

        cloned = clone(model)
        assert cloned.epochs == _TEST_EPOCHS
        assert cloned.monotonicity == [1, 1]
        assert not hasattr(cloned, "params_")

    def test_pipeline_compatibility(self, simple_linear_data):
        """Test usage in sklearn Pipeline."""
        X, y = simple_linear_data

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    JAXMonotonicRegressor(
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

        model = JAXMonotonicRegressor(epochs=5, hidden_dims=(4,))
        param_grid = {"hidden_dims": [(4,), (4, 4)], "learning_rate": [1e-3, 5e-3]}

        # Just test that it runs without error
        grid = GridSearchCV(model, param_grid, cv=2, scoring="r2")
        grid.fit(X, y)

        assert hasattr(grid, "best_params_")


class TestJAXMonotonicStandardization:
    """Test standardization options."""

    def test_no_standardization(self, simple_linear_data):
        """Test with no standardization."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(
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
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            standardize_X=True,
            standardize_y=False,
        )
        model.fit(X, y)

        assert model.scaler_X_ is not None
        assert model.scaler_y_ is None

    def test_y_standardization_only(self, simple_linear_data):
        """Test with only y standardization."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            standardize_X=False,
            standardize_y=True,
        )
        model.fit(X, y)

        assert model.scaler_X_ is None
        assert model.scaler_y_ is not None


class TestJAXMonotonicReproducibility:
    """Test reproducibility with random state."""

    def test_reproducibility(self, simple_linear_data):
        """Test that same random_state gives same results."""
        X, y = simple_linear_data

        model1 = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, random_state=42
        )
        model1.fit(X, y)
        y_pred1 = model1.predict(X)

        model2 = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, random_state=42
        )
        model2.fit(X, y)
        y_pred2 = model2.predict(X)

        np.testing.assert_allclose(y_pred1, y_pred2, rtol=1e-5)

    def test_different_random_states(self, simple_linear_data):
        """Test that different random_states give different results."""
        X, y = simple_linear_data

        model1 = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, random_state=42
        )
        model1.fit(X, y)
        y_pred1 = model1.predict(X)

        model2 = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, random_state=123
        )
        model2.fit(X, y)
        y_pred2 = model2.predict(X)

        # Predictions should be different (not exactly equal)
        assert not np.allclose(y_pred1, y_pred2)


class TestJAXMonotonicActivations:
    """Test different activation functions."""

    def test_softplus_activation(self, simple_linear_data):
        """Test with softplus activation."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, activation="softplus"
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_relu_activation(self, simple_linear_data):
        """Test with relu activation."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, activation="relu"
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestJAXMonotonicParameterization:
    """Test different weight parameterizations."""

    def test_softplus_parameterization(self, monotonic_increasing_data):
        """Test with softplus parameterization."""
        X, y = monotonic_increasing_data
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            nonneg_param="softplus",
            monotonicity=1,
        )
        model.fit(X, y)

        grads = model.predict_gradient(X)
        assert np.all(grads >= -1e-6)

    def test_square_parameterization(self, monotonic_increasing_data):
        """Test with square parameterization."""
        X, y = monotonic_increasing_data
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS,
            hidden_dims=_TEST_HIDDEN_DIMS,
            nonneg_param="square",
            monotonicity=1,
        )
        model.fit(X, y)

        grads = model.predict_gradient(X)
        assert np.all(grads >= -1e-6)


class TestJAXMonotonicEdgeCases:
    """Test edge cases."""

    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        X = np.random.randn(30, 1)
        y = 2 * X.ravel() + 0.1 * np.random.randn(30)

        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, monotonicity=1
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (30,)

    def test_many_features(self):
        """Test with many features."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)

        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, monotonicity=1
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (50,)

    def test_small_dataset(self):
        """Test with small dataset."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = X[:, 0] + X[:, 1]

        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, val_size=0.0
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (20,)

    def test_all_unconstrained(self):
        """Test with all features unconstrained."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2]

        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, monotonicity=0
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (30,)


class TestJAXMonotonicAttributes:
    """Test fitted attributes."""

    def test_n_features_in(self, simple_linear_data):
        """Test n_features_in_ attribute."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        assert model.n_features_in_ == X.shape[1]

    def test_loss_history(self, simple_linear_data):
        """Test loss_history_ attribute."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        assert hasattr(model, "loss_history_")
        assert len(model.loss_history_) == _TEST_EPOCHS

    def test_params_stored(self, simple_linear_data):
        """Test that params_ is stored after fit."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS)
        model.fit(X, y)

        assert hasattr(model, "params_")
        assert "Wx_raw" in model.params_
        assert "Wz_raw" in model.params_
        assert "a_raw" in model.params_
        assert "c_raw" in model.params_
        assert "b_out" in model.params_

    def test_monotonicity_expanded(self, simple_linear_data):
        """Test that monotonicity_ is expanded from scalar."""
        X, y = simple_linear_data
        model = JAXMonotonicRegressor(
            epochs=_TEST_EPOCHS, hidden_dims=_TEST_HIDDEN_DIMS, monotonicity=1
        )
        model.fit(X, y)

        assert hasattr(model, "monotonicity_")
        assert len(model.monotonicity_) == X.shape[1]


class TestJAXMonotonicValidation:
    """Test input validation."""

    def test_invalid_monotonicity_values(self):
        """Test that invalid monotonicity values raise error."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        model = JAXMonotonicRegressor(monotonicity=[1, 2])  # 2 is invalid

        with pytest.raises(ValueError, match="monotonicity values must be"):
            model.fit(X, y)

    def test_wrong_monotonicity_length(self):
        """Test that wrong monotonicity length raises error."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]

        model = JAXMonotonicRegressor(monotonicity=[1, 1, 1])  # 3 != 2 features

        with pytest.raises(ValueError, match="has 3 elements but X has 2"):
            model.fit(X, y)
