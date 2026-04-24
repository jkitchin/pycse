"""
ZENN Regressor with NLL Loss for Calibrated Uncertainty.

This variant uses negative log-likelihood loss to force the S networks
to learn calibrated uncertainty estimates (actual noise variance).

The key insight: In standard ZENN, S networks learn "something" but not
necessarily the true noise level. By using NLL loss:

    NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²

we force σ² (derived from S) to match the actual residual variance.
"""

from typing import Optional, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_array, validate_data

from pycse.sklearn.zenn.estimators.base import ZENNBase
from pycse.sklearn.zenn.utils.thermodynamics import (
    compute_total_helmholtz_energy,
    compute_configuration_probabilities,
)
from pycse.sklearn.zenn.analysis.uncertainty import (
    epistemic_uncertainty,
    aleatoric_uncertainty,
)


class ZENNRegressorNLL(RegressorMixin, ZENNBase):
    """
    ZENN Regressor with Negative Log-Likelihood Loss.

    This variant provides CALIBRATED uncertainty estimates by using NLL loss
    that forces S networks to learn the actual noise variance.

    Parameters
    ----------
    n_configs : int, default=6
        Number of configurations K.

    hidden_dims : tuple, default=(16, 16)
        Hidden layer dimensions.

    kb : float, default=1.0
        Boltzmann constant.

    gamma : float, default=100.0
        Entropy fluctuation scale.

    learning_rate : float, default=0.01
        Learning rate.

    max_epochs : int, default=5000
        Maximum training epochs.

    loss_type : str, default='nll'
        Loss type:
        - 'nll': Pure negative log-likelihood
        - 'hybrid': MSE + NLL weighted combination
        - 'mse': Standard MSE (for comparison)

    nll_weight : float, default=1.0
        Weight for NLL term in hybrid loss.

    mse_weight : float, default=1.0
        Weight for MSE term in hybrid loss.

    min_variance : float, default=1e-6
        Minimum variance to prevent log(0).

    T_default : float, default=1.0
        Default temperature.

    random_state : int or None, default=None
        Random seed.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features.

    Examples
    --------
    >>> from pycse.sklearn.zenn.estimators.regressor_nll import ZENNRegressorNLL
    >>> import numpy as np
    >>>
    >>> # Create noisy data
    >>> x = np.linspace(-2, 2, 100).reshape(-1, 1)
    >>> y_true = x**2
    >>> noise_std = 0.3
    >>> y = y_true + np.random.randn(100, 1) * noise_std
    >>>
    >>> # Train with NLL - uncertainty should be calibrated!
    >>> reg = ZENNRegressorNLL(loss_type='nll', max_epochs=3000)
    >>> reg.fit(x, y.flatten())
    >>>
    >>> # Get calibrated uncertainty
    >>> result = reg.predict_with_uncertainty(x)
    >>> print(f"Learned aleatoric: {result['aleatoric'].mean():.3f}")
    >>> print(f"True noise std: {noise_std}")
    >>> # These should be close!
    """

    _is_classifier = False

    def __init__(
        self,
        n_configs: int = 6,
        hidden_dims: tuple = (16, 16),
        kb: float = 1.0,
        gamma: float = 100.0,
        learning_rate: float = 0.01,
        max_epochs: int = 5000,
        loss_type: str = 'nll',
        nll_weight: float = 1.0,
        mse_weight: float = 1.0,
        min_variance: float = 1e-6,
        T_default: float = 1.0,
        activation: str = 'tanh',
        network_type: str = 'mlp',
        degree: int = 3,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_configs=n_configs,
            hidden_dims=hidden_dims,
            n_temperatures=1,
            kb=kb,
            gamma=gamma,
            learning_rate=learning_rate,
            temperature_lr=0.0,
            max_epochs=max_epochs,
            batch_size=10000,  # Full batch
            early_stopping=False,
            patience=10,
            validation_fraction=0.0,
            convexity_lambda=0.0,
            omega_train=1.0,
            omega_test=1.0,
            activation=activation,
            network_type=network_type,
            degree=degree,
            random_state=random_state,
            verbose=verbose,
        )
        self.loss_type = loss_type
        self.nll_weight = nll_weight
        self.mse_weight = mse_weight
        self.min_variance = min_variance
        self.T_default = T_default

    def _get_n_outputs(self, y: jnp.ndarray) -> int:
        return self.n_configs

    def _compute_variance(
        self,
        S: jnp.ndarray,
        p: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute variance estimate from S networks.

        We use the weighted average of S as variance:
        σ² = Σ_k p(k) * S(k)

        This is the aleatoric uncertainty, now interpreted as variance.
        """
        # Weighted average of S across configurations
        variance = jnp.sum(p * S, axis=-1)
        # Ensure minimum variance for numerical stability
        variance = jnp.maximum(variance, self.min_variance)
        return variance

    def _nll_loss(
        self,
        y_pred: jnp.ndarray,
        y_true: jnp.ndarray,
        variance: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Negative log-likelihood loss for heteroscedastic regression.

        NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²

        This loss naturally balances:
        - Making predictions accurate (minimizing (y - μ)²)
        - Learning correct variance (σ² should match |y - μ|²)

        If σ² is too small: (y - μ)² / σ² term explodes
        If σ² is too large: log(σ²) term explodes
        Optimal: σ² ≈ E[(y - μ)²]
        """
        residual_sq = (y_true - y_pred) ** 2
        nll = 0.5 * jnp.log(variance) + 0.5 * residual_sq / variance
        return jnp.mean(nll)

    def _mse_loss(
        self,
        y_pred: jnp.ndarray,
        y_true: jnp.ndarray,
    ) -> jnp.ndarray:
        """Standard MSE loss."""
        return jnp.mean((y_true - y_pred) ** 2)

    def _compute_loss_with_T(
        self,
        params: Dict[str, Any],
        X: jnp.ndarray,
        y: jnp.ndarray,
        T: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute loss with explicit temperature array."""
        # Forward pass
        outputs = self.model_.apply(params, X, T)
        E, S = outputs['E'], outputs['S']

        # Get configuration probabilities
        p, _ = compute_configuration_probabilities(E, S, T, self.kb, self.gamma)

        # Compute predicted free energy (mean)
        F_pred = compute_total_helmholtz_energy(E, S, T, self.kb, self.gamma)

        # Compute variance from S networks
        variance = self._compute_variance(S, p)

        # Compute loss based on type
        if self.loss_type == 'nll':
            loss = self._nll_loss(F_pred, y, variance)

        elif self.loss_type == 'hybrid':
            mse = self._mse_loss(F_pred, y)
            nll = self._nll_loss(F_pred, y, variance)
            loss = self.mse_weight * mse + self.nll_weight * nll

        elif self.loss_type == 'mse':
            loss = self._mse_loss(F_pred, y)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss

    def _compute_loss(
        self,
        params: Dict[str, Any],
        X: jnp.ndarray,
        y: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute loss (required by base class)."""
        T = temperatures[0] if temperatures.ndim > 0 else temperatures
        return self._compute_loss_with_T(params, X, y, jnp.full((X.shape[0],), T))

    def _predict_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """Internal prediction (required by base class)."""
        T = jnp.full((X.shape[0],), self.T_default)
        outputs = self.model_.apply(self.params_, X, T)
        return compute_total_helmholtz_energy(
            outputs['E'], outputs['S'], T, self.kb, self.gamma
        )

    def fit(self, X, y, T=None):
        """
        Fit the model with NLL loss for calibrated uncertainty.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        T : float or array-like, optional
            Temperature. Default: T_default.

        Returns
        -------
        self
        """
        X, y = validate_data(self, X, y, reset=True, y_numeric=True)
        y = np.asarray(y).flatten()
        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32)

        if T is None:
            T = self.T_default
        if isinstance(T, (int, float)):
            T = jnp.full((X.shape[0],), float(T))
        else:
            T = jnp.array(T, dtype=jnp.float32)

        self._T_train = T
        self.n_features_in_ = X.shape[1]

        # Initialize model
        self._initialize_model(self.n_features_in_, self.n_configs)
        self.temperatures_ = jnp.array([self.T_default])

        # JIT compile training step
        @jax.jit
        def train_step(params, opt_state, X_batch, y_batch, T_batch):
            def loss_fn(p):
                return self._compute_loss_with_T(p, X_batch, y_batch, T_batch)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self.optimizer_.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        # Training loop
        self.history_ = {'loss': []}

        for epoch in range(self.max_epochs):
            self.params_, self.opt_state_, loss = train_step(
                self.params_, self.opt_state_, X, y, T
            )
            self.history_['loss'].append(float(loss))

            if self.verbose and epoch % max(1, self.max_epochs // 10) == 0:
                print(f"Epoch {epoch}: loss={float(loss):.6f}")

        return self

    def predict(self, X, T=None) -> np.ndarray:
        """
        Predict free energy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        T : float, optional
            Temperature.

        Returns
        -------
        F : ndarray of shape (n_samples,)
            Predicted free energy.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = jnp.array(X, dtype=jnp.float32)

        if T is None:
            T = self.T_default
        T_arr = jnp.full((X.shape[0],), float(T))

        outputs = self.model_.apply(self.params_, X, T_arr)
        F = compute_total_helmholtz_energy(
            outputs['E'], outputs['S'], T_arr, self.kb, self.gamma
        )
        return np.array(F)

    def predict_with_uncertainty(
        self,
        X,
        T: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with CALIBRATED uncertainty estimates.

        When trained with NLL loss, the aleatoric uncertainty should
        approximate the true noise standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        T : float, optional
            Temperature.

        Returns
        -------
        dict with:
            - 'prediction': Predicted free energy
            - 'epistemic': Configuration disagreement (model uncertainty)
            - 'aleatoric': Learned noise std (CALIBRATED with NLL loss!)
            - 'variance': Learned variance (aleatoric²)
            - 'total': Combined uncertainty
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        if T is None:
            T = self.T_default
        T_arr = jnp.full((X.shape[0],), float(T))

        outputs = self.model_.apply(self.params_, X, T_arr)
        E, S = outputs['E'], outputs['S']

        # Get configuration probabilities
        p, _ = compute_configuration_probabilities(E, S, T_arr, self.kb, self.gamma)

        # Compute prediction
        F_pred = compute_total_helmholtz_energy(E, S, T_arr, self.kb, self.gamma)

        # Compute uncertainties
        epist = epistemic_uncertainty(E, p)

        # Variance from S networks (calibrated via NLL!)
        variance = self._compute_variance(S, p)
        aleat = jnp.sqrt(variance)  # Convert variance to std

        total = jnp.sqrt(epist ** 2 + aleat ** 2)

        return {
            'prediction': np.array(F_pred),
            'epistemic': np.array(epist),
            'aleatoric': np.array(aleat),
            'variance': np.array(variance),
            'total': np.array(total),
            'config_probs': np.array(p),
            'E_per_config': np.array(E),
            'S_per_config': np.array(S),
        }

    def get_prediction_intervals(
        self,
        X,
        T: Optional[float] = None,
        confidence: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """
        Get prediction intervals using calibrated uncertainty.

        With NLL training, these intervals should have correct coverage!

        Parameters
        ----------
        X : array-like
            Input data.
        T : float, optional
            Temperature.
        confidence : float, default=0.95
            Confidence level.

        Returns
        -------
        dict with 'mean', 'lower', 'upper', 'width'
        """
        from scipy import stats

        result = self.predict_with_uncertainty(X, T)

        z = stats.norm.ppf((1 + confidence) / 2)

        mean = result['prediction']
        std = result['aleatoric']  # Calibrated!

        return {
            'mean': mean,
            'lower': mean - z * std,
            'upper': mean + z * std,
            'width': 2 * z * std,
        }

    def score(self, X, y, T=None) -> float:
        """Return R² score."""
        y_pred = self.predict(X, T)
        y_true = np.asarray(y).flatten()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1 - ss_res / ss_tot

    def get_calibration_metrics(
        self,
        X,
        y,
        T: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute calibration metrics to verify uncertainty quality.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            True values.
        T : float, optional
            Temperature.

        Returns
        -------
        dict with:
            - 'empirical_std': Actual residual std
            - 'learned_std': Mean learned aleatoric (should match empirical!)
            - 'calibration_ratio': learned / empirical (ideal = 1.0)
            - 'coverage_95': Fraction of points within 95% interval
        """
        y_pred = self.predict(X, T)
        y_true = np.asarray(y).flatten()

        residuals = y_true - y_pred
        empirical_std = np.std(residuals)

        result = self.predict_with_uncertainty(X, T)
        learned_std = result['aleatoric'].mean()

        # Check 95% interval coverage
        intervals = self.get_prediction_intervals(X, T, confidence=0.95)
        in_interval = (y_true >= intervals['lower']) & (y_true <= intervals['upper'])
        coverage = np.mean(in_interval)

        return {
            'empirical_std': empirical_std,
            'learned_std': learned_std,
            'calibration_ratio': learned_std / empirical_std,
            'coverage_95': coverage,
        }
