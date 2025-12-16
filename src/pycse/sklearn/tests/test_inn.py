"""Tests for Invertible Neural Network."""

import pytest
import jax
import jax.numpy as np
from pycse.sklearn.inn import InvertibleNN


class TestInvertibleNN:
    """Test suite for InvertibleNN."""

    def test_initialization(self):
        """Test basic initialization."""
        inn = InvertibleNN(n_features=2, n_layers=4, hidden_dims=[32, 32])
        assert inn.n_features == 2
        assert inn.n_layers == 4
        assert inn.hidden_dims == [32, 32]
        assert not inn.is_fitted

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            InvertibleNN(n_features=0)  # Zero features not allowed

        with pytest.raises(ValueError):
            InvertibleNN(n_features=2, n_layers=0)  # Too few layers

        with pytest.raises(ValueError):
            InvertibleNN(n_features=2, hidden_dims=[-1, 32])  # Negative dims

    def test_fit_2d_gaussian(self):
        """Test fitting a 2D Gaussian distribution."""
        # Generate 2D Gaussian data
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (500, 2))

        # Fit model
        inn = InvertibleNN(n_features=2, n_layers=4, hidden_dims=[32, 32], seed=42)
        inn.fit(X, maxiter=500)

        assert inn.is_fitted
        assert hasattr(inn, "params_")
        assert hasattr(inn, "state_")

    def test_forward_inverse_invertibility(self):
        """Test that forward and inverse are truly inverse operations."""
        # Generate and fit data
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 2))

        inn = InvertibleNN(n_features=2, n_layers=4, hidden_dims=[32, 32], seed=42)
        inn.fit(X, maxiter=500)

        # Test invertibility
        Z, _ = inn.forward(X)
        X_reconstructed = inn.inverse(Z)

        # Should reconstruct to high precision
        reconstruction_error = np.max(np.abs(X - X_reconstructed))
        assert reconstruction_error < 1e-6, f"Reconstruction error: {reconstruction_error}"

    def test_log_prob(self):
        """Test log probability computation."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (200, 2))

        inn = InvertibleNN(n_features=2, n_layers=4, hidden_dims=[32, 32], seed=42)
        inn.fit(X, maxiter=500)

        # Compute log probabilities
        log_probs = inn.log_prob(X[:10])

        assert log_probs.shape == (10,)
        assert np.all(np.isfinite(log_probs))
        # Log probs should be reasonable (not too negative)
        assert np.all(log_probs > -100)

    def test_sampling(self):
        """Test sample generation."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (300, 2))

        inn = InvertibleNN(n_features=2, n_layers=4, hidden_dims=[32, 32], seed=42)
        inn.fit(X, maxiter=500)

        # Generate samples
        samples = inn.sample(50, key=jax.random.PRNGKey(123))

        assert samples.shape == (50, 2)
        assert np.all(np.isfinite(samples))

    def test_score_samples(self):
        """Test score_samples (sklearn compatibility)."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 2))

        inn = InvertibleNN(n_features=2, n_layers=4, hidden_dims=[32, 32], seed=42)
        inn.fit(X, maxiter=300)

        scores = inn.score_samples(X[:20])
        assert scores.shape == (20,)

    def test_score(self):
        """Test score method (sklearn compatibility)."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 2))

        inn = InvertibleNN(n_features=2, n_layers=4, hidden_dims=[32, 32], seed=42)
        inn.fit(X, maxiter=300)

        score = inn.score(X)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_normalization(self):
        """Test data normalization."""
        # Create data with specific mean and std
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (200, 2)) * 10 + 50

        inn = InvertibleNN(n_features=2, n_layers=4, hidden_dims=[32, 32], seed=42)
        inn.fit(X, normalize=True, maxiter=300)

        # Check normalization parameters stored
        assert hasattr(inn, "data_mean_")
        assert hasattr(inn, "data_std_")

        # Mean should be close to 50
        assert np.allclose(inn.data_mean_, 50, atol=2)

    def test_unfitted_errors(self):
        """Test that unfitted model raises appropriate errors."""
        inn = InvertibleNN(n_features=2)

        with pytest.raises(RuntimeError):
            inn.forward(np.zeros((1, 2)))

        with pytest.raises(RuntimeError):
            inn.inverse(np.zeros((1, 2)))

        with pytest.raises(RuntimeError):
            inn.log_prob(np.zeros((1, 2)))

        with pytest.raises(RuntimeError):
            inn.sample(10)

    def test_repr_str(self):
        """Test string representations."""
        inn = InvertibleNN(n_features=2, n_layers=4)

        # Before fitting
        repr_str = repr(inn)
        assert "not fitted" in repr_str

        str_str = str(inn)
        assert "not fitted" in str_str

        # After fitting
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 2))
        inn.fit(X, maxiter=100)

        repr_str = repr(inn)
        assert "fitted" in repr_str

        str_str = str(inn)
        assert "fitted" in str_str

    def test_1d_support(self):
        """Test that 1D data is supported via padding."""
        # Create 1D bimodal distribution
        key = jax.random.PRNGKey(42)
        X_1d = np.concatenate(
            [jax.random.normal(key, (100, 1)) - 2, jax.random.normal(key, (100, 1)) + 2]
        )

        # Create and fit 1D model
        inn_1d = InvertibleNN(n_features=1, n_layers=4, hidden_dims=[32, 32], seed=42)
        inn_1d.fit(X_1d, maxiter=300)

        assert inn_1d.is_fitted
        assert inn_1d.n_features == 1

        # Test forward/inverse
        Z, _ = inn_1d.forward(X_1d[:10])
        assert Z.shape == (10, 1)

        X_recon = inn_1d.inverse(Z)
        assert X_recon.shape == (10, 1)

        # Test invertibility
        error = np.max(np.abs(X_1d[:10] - X_recon))
        assert error < 1e-6

        # Test log_prob
        log_probs = inn_1d.log_prob(X_1d[:10])
        assert log_probs.shape == (10,)

        # Test sampling
        samples = inn_1d.sample(50, key=jax.random.PRNGKey(99))
        assert samples.shape == (50, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
