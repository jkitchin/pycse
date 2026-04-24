"""
Tests for ChebyKAN backbone in ZENN.

Tests:
1. ChebyKANLayer shape, JIT, and grad compatibility
2. ZENNRegressor with network_type='kan'
3. ZENNClassifier with network_type='kan'
4. Backward compatibility (default is 'mlp')
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest


class TestChebyKANLayer:
    """Test ChebyKANLayer module."""

    def test_output_shape(self):
        """Verify ChebyKANLayer produces correct output shape."""
        import jax
        import jax.numpy as jnp
        from pycse.sklearn.zenn.networks.kan import ChebyKANLayer

        layer = ChebyKANLayer(out_features=16, degree=3)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((5, 10))
        params = layer.init(key, x)
        y = layer.apply(params, x)
        assert y.shape == (5, 16), f"Expected (5, 16), got {y.shape}"

    def test_jit_compatible(self):
        """Verify ChebyKANLayer works under jax.jit."""
        import jax
        import jax.numpy as jnp
        from pycse.sklearn.zenn.networks.kan import ChebyKANLayer

        layer = ChebyKANLayer(out_features=8, degree=4)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((3, 5))
        params = layer.init(key, x)

        @jax.jit
        def forward(p, x):
            return layer.apply(p, x)

        y = forward(params, x)
        assert y.shape == (3, 8)

    def test_grad_compatible(self):
        """Verify ChebyKANLayer is differentiable."""
        import jax
        import jax.numpy as jnp
        from pycse.sklearn.zenn.networks.kan import ChebyKANLayer

        layer = ChebyKANLayer(out_features=4, degree=3)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 6))
        params = layer.init(key, x)

        def loss_fn(p):
            return jnp.sum(layer.apply(p, x) ** 2)

        grads = jax.grad(loss_fn)(params)
        assert "params" in grads
        assert "cheby_coeffs" in grads["params"]

    @pytest.mark.slow
    def test_different_degrees(self):
        """Verify different polynomial degrees work."""
        import jax
        import jax.numpy as jnp
        from pycse.sklearn.zenn.networks.kan import ChebyKANLayer

        key = jax.random.PRNGKey(0)
        x = jnp.ones((4, 8))

        for degree in [1, 2, 3, 5, 8]:
            layer = ChebyKANLayer(out_features=4, degree=degree)
            params = layer.init(key, x)
            y = layer.apply(params, x)
            assert y.shape == (4, 4), f"Failed for degree={degree}"


class TestConfigurationNetworkKAN:
    """Test ConfigurationNetworkKAN module."""

    def test_output_shape(self):
        """Verify ConfigurationNetworkKAN outputs shape (batch, 1)."""
        import jax
        import jax.numpy as jnp
        from pycse.sklearn.zenn.networks.kan import ConfigurationNetworkKAN

        net = ConfigurationNetworkKAN(hidden_dims=(8, 8), degree=3)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((10, 5))
        params = net.init(key, x)
        y = net.apply(params, x)
        assert y.shape == (10, 1), f"Expected (10, 1), got {y.shape}"


class TestZENNRegressorKAN:
    """Test ZENNRegressor with KAN backbone."""

    @pytest.mark.slow
    def test_kan_fit_predict(self):
        """Test basic fit/predict with network_type='kan'."""
        from pycse.sklearn.zenn import ZENNRegressor

        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(50) * 0.1

        reg = ZENNRegressor(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=100,
            network_type="kan",
            degree=3,
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        y_pred = reg.predict(X)
        assert y_pred.shape == (50,), f"Expected shape (50,), got {y_pred.shape}"

    def test_kan_nll_loss(self):
        """Test KAN regressor with NLL loss."""
        from pycse.sklearn.zenn import ZENNRegressor

        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(50) * 0.2

        reg = ZENNRegressor(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=100,
            loss_type="nll",
            network_type="kan",
            degree=3,
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        result = reg.predict_with_uncertainty(X)
        assert "prediction" in result
        assert "aleatoric" in result
        assert result["prediction"].shape == (50,)
        assert result["aleatoric"].shape == (50,)

    def test_kan_uncertainty(self):
        """Test uncertainty estimates with KAN backbone."""
        from pycse.sklearn.zenn import ZENNRegressor

        np.random.seed(42)
        X = np.linspace(-2, 2, 30).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(30) * 0.1

        reg = ZENNRegressor(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=100,
            network_type="kan",
            degree=3,
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        result = reg.predict_with_uncertainty(X)
        assert "epistemic" in result
        assert "aleatoric" in result
        assert "total" in result
        # Uncertainties should be non-negative
        assert np.all(result["epistemic"] >= 0)
        assert np.all(result["aleatoric"] >= 0)


class TestZENNClassifierKAN:
    """Test ZENNClassifier with KAN backbone."""

    @pytest.mark.slow
    def test_kan_fit_predict(self):
        """Test basic classification with network_type='kan'."""
        from pycse.sklearn.zenn import ZENNClassifier
        from sklearn.datasets import make_classification

        np.random.seed(42)
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )

        clf = ZENNClassifier(
            n_temperatures=1,
            hidden_dims=(16, 16),
            max_epochs=50,
            network_type="kan",
            degree=3,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert y_pred.shape == (100,)
        # All predictions should be valid class labels
        assert set(y_pred).issubset(set(y))

    @pytest.mark.slow
    def test_kan_predict_proba_sums_to_one(self):
        """Test predict_proba sums to 1 with KAN backbone."""
        from pycse.sklearn.zenn import ZENNClassifier
        from sklearn.datasets import make_classification

        np.random.seed(42)
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )

        clf = ZENNClassifier(
            n_temperatures=1,
            hidden_dims=(16, 16),
            max_epochs=50,
            network_type="kan",
            degree=3,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestZENNRegressorNLLKAN:
    """Test ZENNRegressorNLL with KAN backbone."""

    def test_kan_fit_predict(self):
        """Test NLL regressor with KAN backbone."""
        from pycse.sklearn.zenn.estimators.regressor_nll import ZENNRegressorNLL

        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(50) * 0.1

        reg = ZENNRegressorNLL(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=100,
            network_type="kan",
            degree=3,
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        y_pred = reg.predict(X)
        assert y_pred.shape == (50,)

    def test_kan_uncertainty(self):
        """Test NLL regressor uncertainty with KAN backbone."""
        from pycse.sklearn.zenn.estimators.regressor_nll import ZENNRegressorNLL

        np.random.seed(42)
        X = np.linspace(-2, 2, 30).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(30) * 0.2

        reg = ZENNRegressorNLL(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=100,
            network_type="kan",
            degree=3,
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        result = reg.predict_with_uncertainty(X)
        assert "prediction" in result
        assert "aleatoric" in result
        assert "variance" in result


class TestBackwardCompatibility:
    """Test that defaults remain 'mlp' and behavior is unchanged."""

    def test_default_network_type_regressor(self):
        """ZENNRegressor default should be 'mlp'."""
        from pycse.sklearn.zenn import ZENNRegressor

        reg = ZENNRegressor()
        assert reg.network_type == "mlp"

    def test_default_network_type_classifier(self):
        """ZENNClassifier default should be 'mlp'."""
        from pycse.sklearn.zenn import ZENNClassifier

        clf = ZENNClassifier()
        assert clf.network_type == "mlp"

    def test_default_network_type_nll(self):
        """ZENNRegressorNLL default should be 'mlp'."""
        from pycse.sklearn.zenn.estimators.regressor_nll import ZENNRegressorNLL

        reg = ZENNRegressorNLL()
        assert reg.network_type == "mlp"

    def test_mlp_regressor_unchanged(self):
        """MLP regressor should work identically to before."""
        from pycse.sklearn.zenn import ZENNRegressor

        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(50) * 0.1

        reg = ZENNRegressor(
            n_configs=4,
            hidden_dims=(8, 8),
            max_epochs=100,
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        y_pred = reg.predict(X)
        assert y_pred.shape == (50,)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5, f"R² too low: {r2:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
