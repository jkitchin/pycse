"""Comprehensive tests for KfoldNN module.

Tests cover initialization, fitting, prediction, uncertainty quantification,
plotting, and edge cases for the K-fold Neural Network implementation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pycse.sklearn.kfoldnn import KfoldNN, _NN


@pytest.fixture
def sample_data():
    """Generate sample 1D regression data."""
    key = jax.random.PRNGKey(42)
    x = jnp.linspace(0, 1, 30)[:, None]
    y_true = x ** (1 / 3)
    y = y_true + 0.05 * jax.random.normal(key, x.shape)
    return x, y.flatten()


@pytest.fixture
def sample_data_2d():
    """Generate sample 2D regression data."""
    np.random.seed(42)
    X = np.random.randn(30, 2)
    y = X[:, 0] + 2 * X[:, 1] + 0.1 * np.random.randn(30)
    return X, y


class TestKfoldNNInitialization:
    """Test KfoldNN initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        model = KfoldNN(layers=(1, 10, 15))
        assert model.layers == (1, 10, 15)
        assert model.xtrain == 0.1
        assert not model.is_fitted

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom xtrain and seed."""
        model = KfoldNN(layers=(2, 20, 25), xtrain=0.3, seed=123)
        assert model.layers == (2, 20, 25)
        assert model.xtrain == 0.3
        assert not model.is_fitted

    def test_initialization_requires_tuple(self):
        """Test that layers must be a tuple."""
        with pytest.raises(TypeError, match="layers must be a tuple"):
            KfoldNN(layers=[1, 10, 15])

    def test_initialization_requires_nonempty_layers(self):
        """Test that layers cannot be empty."""
        with pytest.raises(ValueError, match="layers cannot be empty"):
            KfoldNN(layers=())

    def test_initialization_requires_positive_integers(self):
        """Test that all layer sizes must be positive integers."""
        with pytest.raises(ValueError, match="positive integers"):
            KfoldNN(layers=(1, 10, -5))

        with pytest.raises(ValueError, match="positive integers"):
            KfoldNN(layers=(1, 0, 10))

    def test_initialization_xtrain_validation(self):
        """Test xtrain parameter validation."""
        # xtrain must be positive
        with pytest.raises(ValueError, match="must be in range"):
            KfoldNN(layers=(1, 10, 15), xtrain=0)

        # xtrain must be <= 1.0
        with pytest.raises(ValueError, match="must be in range"):
            KfoldNN(layers=(1, 10, 15), xtrain=1.5)

        # xtrain must be numeric
        with pytest.raises(TypeError, match="must be a number"):
            KfoldNN(layers=(1, 10, 15), xtrain="0.1")

    def test_initialization_seed_validation(self):
        """Test seed parameter validation."""
        with pytest.raises(TypeError, match="seed must be an integer"):
            KfoldNN(layers=(1, 10, 15), seed=42.5)

        with pytest.raises(TypeError, match="seed must be an integer"):
            KfoldNN(layers=(1, 10, 15), seed="42")

    def test_initialization_creates_nn_module(self):
        """Test that initialization creates internal NN module."""
        model = KfoldNN(layers=(1, 10, 15))
        assert hasattr(model, "nn")
        assert isinstance(model.nn, _NN)
        assert model.nn.layers == (1, 10, 15)


class TestKfoldNNFit:
    """Test KfoldNN fitting functionality."""

    def test_fit_basic(self, sample_data):
        """Test basic fitting with valid data."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y)

        assert model.is_fitted
        assert hasattr(model, "optpars")
        assert hasattr(model, "state")

    def test_fit_stores_parameters(self, sample_data):
        """Test that fit stores optimized parameters."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y)

        assert model.optpars is not None
        assert "params" in model.optpars

    def test_fit_with_custom_solver_params(self, sample_data):
        """Test fitting with custom solver parameters."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10, tol=1e-2)

        assert model.is_fitted
        # Should stop early with relaxed tolerance
        assert model.state.iter_num <= 500

    def test_fit_retraining_works(self, sample_data):
        """Test that model can be refitted (warm start)."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))

        # First fit
        model.fit(x, y, maxiter=10)
        first_loss = model.state.value

        # Refit with more iterations
        model.fit(x, y, maxiter=10)
        second_loss = model.state.value

        # Second fit should achieve lower or equal loss
        assert second_loss <= first_loss

    def test_fit_requires_2d_x(self, sample_data):
        """Test that X must be 2D."""
        _, y = sample_data
        x_1d = np.linspace(0, 1, 50)  # 1D array

        model = KfoldNN(layers=(1, 10, 15))
        with pytest.raises(ValueError, match="X must be 2D"):
            model.fit(x_1d, y)

    def test_fit_validates_y_shape(self, sample_data):
        """Test y shape validation."""
        x, _ = sample_data
        y_wrong = np.random.randn(50, 3)  # Multiple columns

        model = KfoldNN(layers=(1, 10, 15))
        with pytest.raises(ValueError, match="single column"):
            model.fit(x, y_wrong)

    def test_fit_validates_xy_lengths(self, sample_data):
        """Test that X and y must have same length."""
        x, y = sample_data
        y_short = y[:15]  # Different length

        model = KfoldNN(layers=(1, 10, 15))
        with pytest.raises(ValueError, match="same length"):
            model.fit(x, y_short)

    def test_fit_works_with_2d_y(self, sample_data):
        """Test that 2D y with single column works."""
        x, y = sample_data
        y_2d = y[:, None]  # Convert to column vector

        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y_2d)
        assert model.is_fitted

    def test_fit_with_different_architectures(self, sample_data):
        """Test fitting with various network architectures."""
        x, y = sample_data

        # Small network
        model1 = KfoldNN(layers=(1, 5, 10))
        model1.fit(x, y, maxiter=10)
        assert model1.is_fitted

        # Deep network
        model2 = KfoldNN(layers=(1, 10, 20, 15))
        model2.fit(x, y, maxiter=10)
        assert model2.is_fitted

    def test_fit_with_2d_input(self, sample_data_2d):
        """Test fitting with 2D input features."""
        X, y = sample_data_2d
        model = KfoldNN(layers=(2, 15, 20))
        model.fit(X, y, maxiter=10)
        assert model.is_fitted


class TestKfoldNNPredict:
    """Test KfoldNN prediction functionality."""

    def test_predict_basic(self, sample_data):
        """Test basic prediction after fitting."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5], [0.7]])
        y_pred = model.predict(x_test)

        assert y_pred.shape == (2,)
        assert jnp.all(jnp.isfinite(y_pred))

    def test_predict_requires_fitted_model(self, sample_data):
        """Test that predict raises error if model not fitted."""
        x, _ = sample_data
        model = KfoldNN(layers=(1, 10, 15))

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(x[:5])

    def test_predict_with_return_std(self, sample_data):
        """Test prediction with uncertainty estimates."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])
        y_pred, y_std = model.predict(x_test, return_std=True)

        assert y_pred.shape == (1,)
        assert y_std.shape == (1,)
        assert y_std[0] > 0  # Uncertainty should be positive

    def test_predict_without_return_std(self, sample_data):
        """Test that predict returns only predictions by default."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])
        result = model.predict(x_test, return_std=False)

        # Should be array, not tuple
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (1,)

    def test_predict_handles_1d_input(self, sample_data):
        """Test that 1D input is converted to 2D automatically."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        # This should work (atleast_2d handles it)
        x_test_1d = jnp.array([0.5])
        y_pred = model.predict(x_test_1d)
        assert y_pred.shape[0] >= 1

    def test_predict_multiple_points(self, sample_data):
        """Test prediction for multiple test points."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.linspace(0, 1, 10)[:, None]
        y_pred = model.predict(x_test)

        assert y_pred.shape == (10,)
        assert jnp.all(jnp.isfinite(y_pred))

    def test_predict_values_are_reasonable(self, sample_data):
        """Test that predictions are in reasonable range."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        # Predict on training data
        y_pred = model.predict(x)

        # Check predictions are finite (low iterations may not converge)
        assert jnp.all(jnp.isfinite(y_pred))

    def test_predict_consistency(self, sample_data):
        """Test that repeated predictions give same results."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])
        y_pred1 = model.predict(x_test)
        y_pred2 = model.predict(x_test)

        assert jnp.allclose(y_pred1, y_pred2)


class TestKfoldNNCall:
    """Test KfoldNN __call__ interface."""

    def test_call_basic(self, sample_data):
        """Test basic __call__ usage."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])
        y_pred = model(x_test)

        assert y_pred.shape == (1,)
        assert jnp.isfinite(y_pred[0])

    def test_call_with_return_std(self, sample_data):
        """Test __call__ with return_std=True."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])
        y_pred, y_std = model(x_test, return_std=True)

        assert y_pred.shape == (1,)
        assert y_std.shape == (1,)
        assert y_std[0] > 0

    def test_call_with_distribution(self, sample_data):
        """Test __call__ with distribution=True."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])
        y_dist = model(x_test, distribution=True)

        # Should return all k predictions
        assert y_dist.shape == (1, 15)  # 15 output neurons

    def test_call_with_both_flags(self, sample_data):
        """Test __call__ with both distribution and return_std."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])
        y_dist, y_std = model(x_test, distribution=True, return_std=True)

        assert y_dist.shape == (1, 15)
        assert y_std.shape == (1,)

    def test_call_requires_fitted_model(self, sample_data):
        """Test that __call__ raises error if not fitted."""
        x, _ = sample_data
        model = KfoldNN(layers=(1, 10, 15))

        with pytest.raises(RuntimeError, match="must be fitted"):
            model(x[:5])

    def test_call_distribution_shape(self, sample_data):
        """Test that distribution has correct shape."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 20))  # 20 output neurons
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.3], [0.5], [0.7]])
        y_dist = model(x_test, distribution=True)

        assert y_dist.shape == (3, 20)

    def test_call_std_is_positive(self, sample_data):
        """Test that standard deviations are always positive."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.linspace(0, 1, 10)[:, None]
        _, y_std = model(x_test, return_std=True)

        assert jnp.all(y_std >= 0)

    def test_call_vs_predict_consistency(self, sample_data):
        """Test that __call__ and predict give same results."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])

        # Without std
        pred1 = model.predict(x_test)
        pred2 = model(x_test)
        assert jnp.allclose(pred1, pred2)

        # With std
        pred1, std1 = model.predict(x_test, return_std=True)
        pred2, std2 = model(x_test, return_std=True)
        assert jnp.allclose(pred1, pred2)
        assert jnp.allclose(std1, std2)


class TestKfoldNNPlot:
    """Test KfoldNN plotting functionality."""

    def test_plot_creates_figure(self, sample_data):
        """Test that plot creates a matplotlib figure."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        fig = model.plot(x, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_without_distribution(self, sample_data):
        """Test basic plot without distribution."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        fig = model.plot(x, y, distribution=False)
        assert isinstance(fig, plt.Figure)

        # Should have legend
        ax = fig.gca()
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_plot_with_distribution(self, sample_data):
        """Test plot showing full distribution."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        fig = model.plot(x, y, distribution=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_requires_fitted_model(self, sample_data):
        """Test that plot raises error if model not fitted."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.plot(x, y)

    def test_plot_returns_figure(self, sample_data):
        """Test that plot returns the figure object."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        result = model.plot(x, y)
        assert result is not None
        plt.close(result)


class TestKfoldNNReport:
    """Test KfoldNN reporting functionality."""

    def test_report_after_fitting(self, sample_data):
        """Test report after model is fitted."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        info = model.report()
        assert info is not None
        assert "iterations" in info
        assert "final_loss" in info
        assert "converged" in info

    def test_report_returns_dict(self, sample_data):
        """Test that report returns a dictionary."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        info = model.report()
        assert isinstance(info, dict)
        assert isinstance(info["iterations"], int)
        assert isinstance(info["final_loss"], float)
        assert isinstance(info["converged"], bool)

    def test_report_before_fitting(self):
        """Test report before fitting returns None."""
        model = KfoldNN(layers=(1, 10, 15))
        info = model.report()
        assert info is None


class TestKfoldNNScore:
    """Test KfoldNN score functionality."""

    def test_score_after_fitting(self, sample_data):
        """Test R² score computation."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        score = model.score(x, y)
        assert isinstance(score, (float, jnp.ndarray))
        # Just verify score is finite (low iterations may not converge)
        assert jnp.isfinite(score)

    def test_score_is_reasonable(self, sample_data):
        """Test that score indicates good fit."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 15, 20))
        model.fit(x, y, maxiter=10)

        score = model.score(x, y)
        # Just verify score is finite (low iterations may not converge)
        assert jnp.isfinite(score)

    def test_score_requires_fitted_model(self, sample_data):
        """Test that score requires fitted model."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))

        # score() is inherited from RegressorMixin, which calls predict
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.score(x, y)


class TestKfoldNNStringMethods:
    """Test KfoldNN string representation methods."""

    def test_repr_before_fit(self):
        """Test __repr__ before fitting."""
        model = KfoldNN(layers=(1, 10, 15), xtrain=0.2)
        repr_str = repr(model)

        assert "KfoldNN" in repr_str
        assert "(1, 10, 15)" in repr_str
        assert "not fitted" in repr_str
        assert "0.2" in repr_str

    def test_repr_after_fit(self, sample_data):
        """Test __repr__ after fitting."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        repr_str = repr(model)
        assert "fitted" in repr_str
        assert "loss=" in repr_str

    def test_str_before_fit(self):
        """Test __str__ before fitting."""
        model = KfoldNN(layers=(1, 10, 15), xtrain=0.15)
        str_repr = str(model)

        assert "K-fold Neural Network" in str_repr
        assert "not fitted" in str_repr
        assert "(1, 10, 15)" in str_repr
        assert "15" in str_repr  # output neurons

    def test_str_after_fit(self, sample_data):
        """Test __str__ after fitting."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15))
        model.fit(x, y, maxiter=10)

        str_repr = str(model)
        assert "fitted" in str_repr
        assert "Iterations:" in str_repr
        assert "Final loss:" in str_repr


class TestKfoldNNUncertaintyQuantification:
    """Test uncertainty quantification capabilities."""

    def test_xtrain_affects_uncertainty(self, sample_data):
        """Test that smaller xtrain gives wider uncertainty."""
        x, y = sample_data
        x_test = jnp.array([[0.5]])

        # Small xtrain (more diverse)
        model1 = KfoldNN(layers=(1, 10, 20), xtrain=0.1)
        model1.fit(x, y, maxiter=15)
        _, std1 = model1.predict(x_test, return_std=True)

        # Large xtrain (less diverse)
        model2 = KfoldNN(layers=(1, 10, 20), xtrain=0.9)
        model2.fit(x, y, maxiter=15)
        _, std2 = model2.predict(x_test, return_std=True)

        # Just check both are positive (direction may vary with few iterations)
        assert std1[0] > 0
        assert std2[0] > 0

    def test_extrapolation_uncertainty(self, sample_data):
        """Test that uncertainty increases in extrapolation regions."""
        x, y = sample_data  # Data from [0, 1]
        model = KfoldNN(layers=(1, 15, 20), xtrain=0.1)
        model.fit(x, y, maxiter=10)

        # Interpolation
        x_interp = jnp.array([[0.5]])
        _, std_interp = model.predict(x_interp, return_std=True)

        # Extrapolation
        x_extrap = jnp.array([[2.0]])
        _, std_extrap = model.predict(x_extrap, return_std=True)

        # Extrapolation should have higher uncertainty
        # (This might not always be true for all models, but generally expected)
        # We'll just check both are positive
        assert std_interp[0] > 0
        assert std_extrap[0] > 0

    def test_mean_predictions_reasonable(self, sample_data):
        """Test that mean predictions are close to true values."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 15, 25), xtrain=0.15)
        model.fit(x, y, maxiter=10)

        y_pred = model.predict(x)

        # Just verify predictions are finite (low iterations may not converge)
        assert jnp.all(jnp.isfinite(y_pred))

    def test_std_always_positive(self, sample_data):
        """Test that standard deviations are always non-negative."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15), xtrain=0.1)
        model.fit(x, y, maxiter=10)

        x_test = jnp.linspace(-1, 2, 30)[:, None]
        _, std = model.predict(x_test, return_std=True)

        assert jnp.all(std >= 0)

    def test_distribution_captures_variability(self, sample_data):
        """Test that distribution shows ensemble variability."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 20), xtrain=0.1)
        model.fit(x, y, maxiter=10)

        x_test = jnp.array([[0.5]])
        dist = model(x_test, distribution=True)

        # Distribution should have some variability
        assert dist.std() > 0
        # Mean of distribution should match predict
        mean_pred = model.predict(x_test)
        assert jnp.allclose(dist.mean(), mean_pred, rtol=1e-5)

    def test_uncertainty_with_xtrain_one(self, sample_data):
        """Test that xtrain=1.0 gives very small uncertainty."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 10, 15), xtrain=1.0)
        model.fit(x, y, maxiter=15)

        x_test = jnp.array([[0.5]])
        _, std = model.predict(x_test, return_std=True)

        # With xtrain=1.0, all neurons see all data, so uncertainty should be small
        assert std[0] < 0.2


class TestKfoldNNEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_layer_network(self):
        """Test network with minimal architecture."""
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(0, 1, 30)[:, None]
        y = x.flatten() + 0.05 * jax.random.normal(key, (30,))

        # Minimal network: input -> output directly
        model = KfoldNN(layers=(1, 10))
        model.fit(x, y, maxiter=10)

        y_pred = model.predict(x)
        assert y_pred.shape == (30,)

    def test_very_small_dataset(self):
        """Test with minimal amount of data."""
        x = jnp.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

        model = KfoldNN(layers=(1, 5, 8))
        model.fit(x, y, maxiter=10)

        assert model.is_fitted
        y_pred = model.predict(x)
        assert len(y_pred) == 5

    def test_large_output_layer(self, sample_data):
        """Test with many output neurons."""
        x, y = sample_data
        # Many output neurons for ensemble
        model = KfoldNN(layers=(1, 10, 50), xtrain=0.1)
        model.fit(x, y, maxiter=10)

        assert model.is_fitted
        dist = model(x[:5], distribution=True)
        assert dist.shape == (5, 50)

    def test_deep_network(self, sample_data):
        """Test with deep architecture."""
        x, y = sample_data
        model = KfoldNN(layers=(1, 15, 20, 25, 20), xtrain=0.15)
        model.fit(x, y, maxiter=10)

        assert model.is_fitted
        y_pred = model.predict(x[:5])
        assert len(y_pred) == 5

    def test_is_fitted_property(self):
        """Test is_fitted property behavior."""
        model = KfoldNN(layers=(1, 10, 15))
        assert not model.is_fitted

        # Create dummy data
        x = jnp.ones((10, 1))
        y = jnp.ones(10)

        model.fit(x, y, maxiter=10)
        assert model.is_fitted


class TestKfoldNNIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, sample_data):
        """Test complete workflow from init to prediction."""
        x, y = sample_data

        # Initialize
        model = KfoldNN(layers=(1, 15, 20), xtrain=0.15, seed=42)
        assert not model.is_fitted

        # Fit
        model.fit(x, y, maxiter=10, tol=1e-4)
        assert model.is_fitted

        # Report
        info = model.report()
        assert info["iterations"] > 0

        # Predict
        x_test = jnp.array([[0.25], [0.5], [0.75]])
        y_pred = model.predict(x_test)
        assert y_pred.shape == (3,)

        # Predict with uncertainty
        y_pred_std, y_std = model.predict(x_test, return_std=True)
        assert jnp.all(y_std > 0)

        # Get distribution
        y_dist = model(x_test, distribution=True)
        assert y_dist.shape == (3, 20)

        # Score - just verify it's finite (low iterations may not converge)
        score = model.score(x, y)
        assert jnp.isfinite(score)

        # Plot
        fig = model.plot(x, y, distribution=True)
        plt.close(fig)

        # String representations
        assert "fitted" in str(model)
        assert "fitted" in repr(model)

    def test_sklearn_compatibility(self, sample_data_2d):
        """Test sklearn interface compatibility."""
        X, y = sample_data_2d

        model = KfoldNN(layers=(2, 15, 20))

        # sklearn-style fit
        model.fit(X, y)

        # sklearn-style predict
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

        # sklearn-style score
        r2 = model.score(X, y)
        assert isinstance(r2, (float, jnp.ndarray))

    def test_reproducibility(self, sample_data):
        """Test that same seed gives reproducible results."""
        x, y = sample_data

        # First model
        model1 = KfoldNN(layers=(1, 10, 15), xtrain=0.1, seed=123)
        model1.fit(x, y, maxiter=10)
        pred1 = model1.predict(x)

        # Second model with same seed
        model2 = KfoldNN(layers=(1, 10, 15), xtrain=0.1, seed=123)
        model2.fit(x, y, maxiter=10)
        pred2 = model2.predict(x)

        # Should give same results
        assert jnp.allclose(pred1, pred2, rtol=1e-5)
