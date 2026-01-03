"""Tests for JAXICNNRegressor - Input Convex Neural Network."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pycse.sklearn.jax_icnn import JAXICNNRegressor


@pytest.fixture
def quadratic_data():
    """Generate data from a convex quadratic function."""
    np.random.seed(42)
    X = np.random.randn(100, 2) * 2
    # y = x1^2 + x2^2 (convex)
    y = np.sum(X ** 2, axis=1) + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def linear_data():
    """Generate simple linear data."""
    np.random.seed(42)
    X = np.random.randn(80, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.1 * np.random.randn(80)
    return X, y


@pytest.fixture
def high_dim_data():
    """Generate higher dimensional data."""
    np.random.seed(42)
    X = np.random.randn(150, 5)
    # Convex function
    y = np.sum(X ** 2, axis=1) + np.sum(np.abs(X), axis=1)
    return X, y


class TestJAXICNNInitialization:
    """Test initialization of JAXICNNRegressor."""

    def test_default_initialization(self):
        """Test default parameter values."""
        model = JAXICNNRegressor()

        assert model.hidden_dims == (32, 32)
        assert model.activation == "softplus"
        assert model.nonneg_param == "softplus"
        assert model.learning_rate == 5e-3
        assert model.weight_decay == 0.0
        assert model.epochs == 500
        assert model.batch_size == 32
        assert model.standardize_X is True
        assert model.standardize_y is True
        assert model.strong_convexity_mu == 0.0
        assert model.random_state == 42
        assert model.verbose is False

    def test_custom_initialization(self):
        """Test custom parameter values."""
        model = JAXICNNRegressor(
            hidden_dims=(64, 32, 16),
            activation="relu",
            nonneg_param="square",
            learning_rate=5e-4,
            weight_decay=1e-4,
            epochs=1000,
            batch_size=64,
            standardize_X=False,
            standardize_y=False,
            strong_convexity_mu=0.1,
            random_state=123,
            verbose=True,
        )

        assert model.hidden_dims == (64, 32, 16)
        assert model.activation == "relu"
        assert model.nonneg_param == "square"
        assert model.learning_rate == 5e-4
        assert model.weight_decay == 1e-4
        assert model.epochs == 1000
        assert model.batch_size == 64
        assert model.standardize_X is False
        assert model.standardize_y is False
        assert model.strong_convexity_mu == 0.1
        assert model.random_state == 123
        assert model.verbose is True


class TestJAXICNNBasicFunctionality:
    """Test basic fit/predict functionality."""

    def test_fit_returns_self(self, quadratic_data):
        """Test that fit returns self."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=10)
        result = model.fit(X, y)
        assert result is model

    def test_fit_predict_basic(self, quadratic_data):
        """Test basic fit and predict cycle."""
        X, y = quadratic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = JAXICNNRegressor(epochs=50, hidden_dims=(16, 16))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))

    def test_predict_shape_single_sample(self, quadratic_data):
        """Test prediction on single sample."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=10)
        model.fit(X, y)

        # Single sample as 1D array
        y_pred = model.predict(X[0:1])
        assert y_pred.shape == (1,)

    def test_predict_shape_batch(self, quadratic_data):
        """Test prediction on batch."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=10)
        model.fit(X, y)

        y_pred = model.predict(X[:20])
        assert y_pred.shape == (20,)

    def test_fit_with_1d_y(self, quadratic_data):
        """Test fit works with 1D y array."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=10)
        model.fit(X, y.ravel())
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)

    def test_fit_with_2d_y(self, quadratic_data):
        """Test fit works with 2D y array."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=10)
        model.fit(X, y.reshape(-1, 1))
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)


class TestJAXICNNGradients:
    """Test gradient computation."""

    def test_predict_gradient_shape(self, quadratic_data):
        """Test gradient output shape."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=50)
        model.fit(X, y)

        grad = model.predict_gradient(X[:10])

        assert grad.shape == (10, X.shape[1])
        assert np.all(np.isfinite(grad))

    def test_predict_gradient_single_sample(self, quadratic_data):
        """Test gradient on single sample."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=50)
        model.fit(X, y)

        grad = model.predict_gradient(X[0:1])
        assert grad.shape == (1, X.shape[1])

    def test_predict_with_grad_consistency(self, quadratic_data):
        """Test predict_with_grad returns same as separate calls."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=50)
        model.fit(X, y)

        y_pred1 = model.predict(X[:5])
        grad1 = model.predict_gradient(X[:5])

        y_pred2, grad2 = model.predict_with_grad(X[:5])

        np.testing.assert_allclose(y_pred1, y_pred2, rtol=1e-5)
        np.testing.assert_allclose(grad1, grad2, rtol=1e-5)


class TestJAXICNNConvexity:
    """Test convexity properties."""

    def test_convexity_midpoint(self, quadratic_data):
        """Test convexity: f((x+y)/2) <= (f(x) + f(y))/2."""
        X, y = quadratic_data
        model = JAXICNNRegressor(
            epochs=100,
            hidden_dims=(32, 32),
            standardize_X=False,  # Easier to verify convexity
            standardize_y=False,
        )
        model.fit(X, y)

        # Generate random pairs
        np.random.seed(123)
        n_pairs = 50
        x1 = np.random.randn(n_pairs, X.shape[1])
        x2 = np.random.randn(n_pairs, X.shape[1])
        x_mid = (x1 + x2) / 2

        f_x1 = model.predict(x1)
        f_x2 = model.predict(x2)
        f_mid = model.predict(x_mid)

        # Check convexity: f(mid) <= (f(x1) + f(x2))/2
        avg = (f_x1 + f_x2) / 2

        # Allow small numerical tolerance
        violations = np.sum(f_mid > avg + 1e-5)
        assert violations == 0, f"Convexity violated {violations}/{n_pairs} times"

    def test_convexity_interpolation(self, linear_data):
        """Test convexity along random lines."""
        X, y = linear_data
        model = JAXICNNRegressor(
            epochs=100,
            hidden_dims=(16, 16),
            standardize_X=False,
            standardize_y=False,
        )
        model.fit(X, y)

        # Test along random lines
        np.random.seed(456)
        n_lines = 20
        n_points = 10

        for _ in range(n_lines):
            x1 = np.random.randn(1, X.shape[1])
            x2 = np.random.randn(1, X.shape[1])

            # Points along line
            t = np.linspace(0, 1, n_points).reshape(-1, 1)
            points = (1 - t) * x1 + t * x2

            f_values = model.predict(points)

            # For convex function, all points should be below or on the line
            # connecting endpoints
            f_line = (1 - t.ravel()) * f_values[0] + t.ravel() * f_values[-1]

            # Allow small tolerance
            assert np.all(f_values <= f_line + 1e-4), "Convexity violated along line"


class TestJAXICNNStrongConvexity:
    """Test strong convexity."""

    def test_strong_convexity_prediction(self, linear_data):
        """Test that strong_convexity_mu adds quadratic term to output."""
        X, y = linear_data
        mu = 0.5

        model_base = JAXICNNRegressor(
            epochs=50,
            strong_convexity_mu=0.0,
            random_state=42,
        )
        model_base.fit(X, y)

        model_sc = JAXICNNRegressor(
            epochs=50,
            strong_convexity_mu=mu,
            random_state=42,
        )
        model_sc.fit(X, y)

        # The difference should be (mu/2) * ||x||^2
        y_base = model_base.predict(X[:10])
        y_sc = model_sc.predict(X[:10])

        expected_diff = 0.5 * mu * np.sum(X[:10] ** 2, axis=1)

        np.testing.assert_allclose(y_sc - y_base, expected_diff, rtol=1e-5)

    def test_strong_convexity_gradient(self, linear_data):
        """Test that strong convexity adds mu*x to gradient."""
        X, y = linear_data
        mu = 0.3

        model_base = JAXICNNRegressor(
            epochs=50,
            strong_convexity_mu=0.0,
            random_state=42,
        )
        model_base.fit(X, y)

        model_sc = JAXICNNRegressor(
            epochs=50,
            strong_convexity_mu=mu,
            random_state=42,
        )
        model_sc.fit(X, y)

        # Gradient difference should be mu * x
        grad_base = model_base.predict_gradient(X[:10])
        grad_sc = model_sc.predict_gradient(X[:10])

        expected_diff = mu * X[:10]

        np.testing.assert_allclose(grad_sc - grad_base, expected_diff, rtol=1e-5)


class TestJAXICNNStandardization:
    """Test standardization options."""

    def test_with_standardization(self, quadratic_data):
        """Test with both X and y standardization."""
        X, y = quadratic_data
        model = JAXICNNRegressor(
            epochs=50,
            standardize_X=True,
            standardize_y=True,
        )
        model.fit(X, y)

        assert model.scaler_X_ is not None
        assert model.scaler_y_ is not None

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_without_standardization(self, quadratic_data):
        """Test without standardization."""
        X, y = quadratic_data
        model = JAXICNNRegressor(
            epochs=50,
            standardize_X=False,
            standardize_y=False,
        )
        model.fit(X, y)

        assert model.scaler_X_ is None
        assert model.scaler_y_ is None

        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_partial_standardization(self, quadratic_data):
        """Test with only X standardization."""
        X, y = quadratic_data
        model = JAXICNNRegressor(
            epochs=50,
            standardize_X=True,
            standardize_y=False,
        )
        model.fit(X, y)

        assert model.scaler_X_ is not None
        assert model.scaler_y_ is None


class TestJAXICNNActivations:
    """Test different activation functions."""

    def test_softplus_activation(self, quadratic_data):
        """Test with softplus activation (default)."""
        X, y = quadratic_data
        model = JAXICNNRegressor(
            epochs=50,
            activation="softplus",
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_relu_activation(self, quadratic_data):
        """Test with relu activation."""
        X, y = quadratic_data
        model = JAXICNNRegressor(
            epochs=50,
            activation="relu",
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestJAXICNNNonnegParam:
    """Test different nonnegativity parameterizations."""

    def test_softplus_param(self, quadratic_data):
        """Test with softplus parameterization (default)."""
        X, y = quadratic_data
        model = JAXICNNRegressor(
            epochs=50,
            nonneg_param="softplus",
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_square_param(self, quadratic_data):
        """Test with square parameterization."""
        X, y = quadratic_data
        model = JAXICNNRegressor(
            epochs=50,
            nonneg_param="square",
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestJAXICNNScore:
    """Test score method."""

    def test_score_returns_float(self, quadratic_data):
        """Test score method returns a float."""
        X, y = quadratic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = JAXICNNRegressor(epochs=100, hidden_dims=(32, 32))
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_score_positive_for_good_fit(self, quadratic_data):
        """Test score is positive for reasonable fit."""
        X, y = quadratic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Use default learning rate (5e-3) which works well for ICNN
        model = JAXICNNRegressor(epochs=500, hidden_dims=(32, 32))
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        # ICNN should achieve good fit on convex quadratic data
        assert score > 0.8, f"Expected R² > 0.8, got {score}"


class TestJAXICNNSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_get_params(self):
        """Test get_params method."""
        model = JAXICNNRegressor(hidden_dims=(64,), epochs=100)
        params = model.get_params()

        assert params["hidden_dims"] == (64,)
        assert params["epochs"] == 100

    def test_set_params(self):
        """Test set_params method."""
        model = JAXICNNRegressor()
        model.set_params(epochs=200, learning_rate=1e-4)

        assert model.epochs == 200
        assert model.learning_rate == 1e-4

    def test_pipeline_compatibility(self, quadratic_data):
        """Test compatibility with sklearn Pipeline."""
        X, y = quadratic_data

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("icnn", JAXICNNRegressor(epochs=20, standardize_X=False)),
        ])

        pipe.fit(X, y)
        y_pred = pipe.predict(X)

        assert y_pred.shape == (len(X),)

    def test_gridsearchcv_compatibility(self, quadratic_data):
        """Test compatibility with GridSearchCV."""
        X, y = quadratic_data

        model = JAXICNNRegressor(epochs=20)
        param_grid = {
            "hidden_dims": [(16,), (16, 16)],
            "learning_rate": [1e-3, 1e-2],
        }

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=2,
            scoring="r2",
        )
        grid_search.fit(X, y)

        assert hasattr(grid_search, "best_params_")
        assert hasattr(grid_search, "best_score_")


class TestJAXICNNReproducibility:
    """Test reproducibility."""

    def test_same_seed_same_results(self, quadratic_data):
        """Test that same random_state gives same results."""
        X, y = quadratic_data

        model1 = JAXICNNRegressor(epochs=50, random_state=123)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = JAXICNNRegressor(epochs=50, random_state=123)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)

    def test_different_seed_different_results(self, quadratic_data):
        """Test that different random_state gives different results."""
        X, y = quadratic_data

        model1 = JAXICNNRegressor(epochs=50, random_state=123)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = JAXICNNRegressor(epochs=50, random_state=456)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        # Results should be different (not exactly equal)
        assert not np.allclose(pred1, pred2, rtol=1e-10)


class TestJAXICNNLossHistory:
    """Test training history."""

    def test_loss_history_populated(self, quadratic_data):
        """Test that loss_history_ is populated after fit."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=50)
        model.fit(X, y)

        assert hasattr(model, "loss_history_")
        assert len(model.loss_history_) == 50
        assert all(np.isfinite(loss) for loss in model.loss_history_)

    def test_loss_decreases(self, quadratic_data):
        """Test that loss generally decreases during training."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=100)
        model.fit(X, y)

        # Compare first 10% to last 10% of training
        early_loss = np.mean(model.loss_history_[:10])
        late_loss = np.mean(model.loss_history_[-10:])

        assert late_loss < early_loss, "Loss should decrease during training"


class TestJAXICNNEdgeCases:
    """Test edge cases."""

    def test_single_hidden_layer(self, linear_data):
        """Test with single hidden layer."""
        X, y = linear_data
        model = JAXICNNRegressor(hidden_dims=(32,), epochs=50)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)

    def test_deep_network(self, linear_data):
        """Test with deep network."""
        X, y = linear_data
        model = JAXICNNRegressor(hidden_dims=(16, 16, 16, 16), epochs=50)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)

    def test_high_dimensional_input(self, high_dim_data):
        """Test with higher dimensional input."""
        X, y = high_dim_data
        model = JAXICNNRegressor(hidden_dims=(32, 32), epochs=50)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (len(X),)

    def test_small_batch_size(self, quadratic_data):
        """Test with small batch size."""
        X, y = quadratic_data
        model = JAXICNNRegressor(batch_size=8, epochs=50)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_large_batch_size(self, quadratic_data):
        """Test with batch size larger than dataset."""
        X, y = quadratic_data
        model = JAXICNNRegressor(batch_size=1000, epochs=50)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestJAXICNNWeightDecay:
    """Test weight decay regularization."""

    def test_weight_decay_zero(self, quadratic_data):
        """Test with no weight decay."""
        X, y = quadratic_data
        model = JAXICNNRegressor(weight_decay=0.0, epochs=50)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_weight_decay_positive(self, quadratic_data):
        """Test with positive weight decay."""
        X, y = quadratic_data
        model = JAXICNNRegressor(weight_decay=1e-4, epochs=50)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert np.all(np.isfinite(y_pred))


class TestJAXICNNAttributes:
    """Test fitted attributes."""

    def test_n_features_in(self, quadratic_data):
        """Test n_features_in_ attribute."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=10)
        model.fit(X, y)

        assert model.n_features_in_ == X.shape[1]

    def test_params_attribute(self, quadratic_data):
        """Test params_ attribute exists after fit."""
        X, y = quadratic_data
        model = JAXICNNRegressor(epochs=10)
        model.fit(X, y)

        assert hasattr(model, "params_")
        assert "Wx" in model.params_
        assert "Wz_raw" in model.params_
        assert "b" in model.params_
        assert "a_raw" in model.params_
        assert "c" in model.params_
        assert "b_out" in model.params_


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
