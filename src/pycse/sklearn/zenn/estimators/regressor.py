"""
ZENN Regressor for energy landscape reconstruction.

Implements both MSE and NLL (negative log-likelihood) losses.
NLL is the default as it provides calibrated uncertainty estimates.
"""

from typing import Optional, Dict, Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_array, validate_data

from pycse.sklearn.zenn.estimators.base import ZENNBase
from pycse.sklearn.zenn.losses.jensen_shannon import energy_landscape_loss, mse_loss
from pycse.sklearn.zenn.losses.constraints import (
    convexity_penalty_from_ES,
    combined_regularization,
)
from pycse.sklearn.zenn.utils.thermodynamics import (
    compute_total_helmholtz_energy,
    compute_configuration_probabilities,
)


class ZENNRegressor(RegressorMixin, ZENNBase):
    """
    Zentropy-Enhanced Neural Network Regressor.

    A thermodynamics-inspired regressor for energy landscape reconstruction.
    Uses zentropy theory to model Helmholtz energy surfaces with multiple
    configurations, enabling robust high-order derivative prediction and
    critical point detection.

    **Default loss is NLL** which provides calibrated uncertainty estimates.
    The S networks learn actual noise variance, enabling accurate prediction
    intervals and uncertainty quantification.

    Parameters
    ----------
    n_configs : int, default=6
        Number of configurations K. More configurations capture more
        complex energy landscapes (paper uses 6-12).

    hidden_dims : tuple, default=(16, 16)
        Hidden layer dimensions. Larger networks for complex landscapes.

    kb : float, default=1.0
        Boltzmann constant. Adjust for energy scaling.

    gamma : float, default=100.0
        Entropy fluctuation scale parameter.

    learning_rate : float, default=0.01
        Learning rate for model parameters.

    max_epochs : int, default=5000
        Maximum training epochs. Energy landscapes need many epochs.

    batch_size : int, default=None
        Mini-batch size. None for full-batch (recommended for small datasets).

    loss_type : str, default='nll'
        Loss type:
        - 'nll': Negative log-likelihood (DEFAULT) - calibrated uncertainty
        - 'mse': Mean squared error - faster but uncalibrated uncertainty
        - 'hybrid': Weighted combination of MSE + NLL
        - 'js': Jensen-Shannon divergence for distribution matching

    nll_weight : float, default=1.0
        Weight for NLL term in hybrid loss.

    mse_weight : float, default=1.0
        Weight for MSE term in hybrid loss.

    min_variance : float, default=1e-6
        Minimum variance to prevent numerical issues in NLL loss.

    convexity_lambda : float, default=0.0
        Weight for convexity constraint (Eq. 20). Set > 0 to enforce
        physical consistency. Note: often not needed with NLL loss.

    smoothness_lambda : float, default=0.0
        Weight for smoothness regularization.

    T_default : float, default=1.0
        Default temperature for predictions.

    random_state : int or None, default=None
        Random seed.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features.

    Notes
    -----
    **Why NLL is the default:**

    With MSE loss, S networks learn "something" related to entropy but not
    necessarily calibrated noise estimates. With NLL loss:

        NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²

    The S networks are forced to learn σ² ≈ E[(y - μ)²], the actual residual
    variance. This gives:
    - Calibrated prediction intervals with correct coverage
    - Meaningful aleatoric uncertainty estimates
    - Better scientific interpretation of the learned model

    Examples
    --------
    >>> from zenn import ZENNRegressor
    >>> import numpy as np
    >>> # Create noisy double-well potential data
    >>> x = np.linspace(-2, 2, 100).reshape(-1, 1)
    >>> F_true = x**4 - 2*x**2
    >>> noise_std = 0.1
    >>> F = F_true + np.random.randn(100, 1) * noise_std
    >>> reg = ZENNRegressor(n_configs=6, max_epochs=3000)
    >>> reg.fit(x, F.flatten())
    >>> # Get calibrated uncertainty
    >>> result = reg.predict_with_uncertainty(x)
    >>> print(f"Learned aleatoric: {result['aleatoric'].mean():.3f}")
    >>> print(f"True noise std: {noise_std}")  # These should be close!
    """

    _is_classifier = False
    _estimator_type = "regressor"

    def __init__(
        self,
        n_configs: int = 6,
        hidden_dims: tuple = (16, 16),
        kb: float = 1.0,
        gamma: float = 100.0,
        learning_rate: float = 0.01,
        max_epochs: int = 5000,
        batch_size: int = 10000,
        loss_type: str = "nll",  # NLL is now default!
        nll_weight: float = 1.0,
        mse_weight: float = 1.0,
        min_variance: float = 1e-6,
        convexity_lambda: float = 0.0,
        smoothness_lambda: float = 0.0,
        T_default: float = 1.0,
        activation: str = "tanh",
        network_type: str = "mlp",
        degree: int = 3,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_configs=n_configs,
            hidden_dims=hidden_dims,
            n_temperatures=1,  # Usually fixed T for regression
            kb=kb,
            gamma=gamma,
            learning_rate=learning_rate,
            temperature_lr=0.0,
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping=False,
            patience=10,
            validation_fraction=0.0,
            convexity_lambda=convexity_lambda,
            omega_train=1.0,
            omega_test=1.0,
            activation=activation,
            network_type=network_type,
            degree=degree,
            random_state=random_state,
            verbose=verbose,
        )
        self.smoothness_lambda = smoothness_lambda
        self.loss_type = loss_type
        self.nll_weight = nll_weight
        self.mse_weight = mse_weight
        self.min_variance = min_variance
        self.T_default = T_default

    def _validate_data(self, X, y):
        """Validate input data."""
        X, y = validate_data(self, X, y, reset=True, y_numeric=True)
        y = np.asarray(y).flatten()
        return X, y

    def _get_n_outputs(self, y: jnp.ndarray) -> int:
        """For regression, n_outputs is n_configs."""
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

        This is the aleatoric uncertainty, interpreted as variance when
        using NLL loss.
        """
        variance = jnp.sum(p * S, axis=-1)
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

    def fit(self, X, y, T=None):
        """
        Fit the ZENN model to energy landscape data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates (e.g., volume, position).

        y : array-like of shape (n_samples,)
            Target Helmholtz energy values.

        T : array-like of shape (n_samples,) or float, optional
            Temperature values. If float, uses same T for all samples.
            If None, uses T_default.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y)
        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32)

        # Handle temperature
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

        # Override temperatures with single value
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

        import optax

        # Training loop (full-batch for energy landscapes)
        self.history_ = {"loss": []}

        for epoch in range(self.max_epochs):
            self.params_, self.opt_state_, loss = train_step(
                self.params_, self.opt_state_, X, y, T
            )
            self.history_["loss"].append(float(loss))

            if self.verbose and epoch % max(1, self.max_epochs // 10) == 0:
                print(f"Epoch {epoch}: loss={float(loss):.6f}")

        return self

    def _compute_loss(
        self,
        params: Dict[str, Any],
        X: jnp.ndarray,
        y: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute loss for training (used by base class)."""
        T = temperatures[0] if temperatures.ndim > 0 else temperatures
        return self._compute_loss_with_T(params, X, y, jnp.full((X.shape[0],), T))

    def _compute_loss_with_T(
        self,
        params: Dict[str, Any],
        X: jnp.ndarray,
        y: jnp.ndarray,
        T: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute loss with explicit temperature array."""
        outputs = self.model_.apply(params, X, T)
        E, S = outputs["E"], outputs["S"]

        # Get configuration probabilities (needed for NLL)
        p, _ = compute_configuration_probabilities(E, S, T, self.kb, self.gamma)

        # Compute predicted total Helmholtz energy
        F_pred = compute_total_helmholtz_energy(E, S, T, self.kb, self.gamma)

        # Compute variance from S networks (for NLL)
        variance = self._compute_variance(S, p)

        # Main loss based on type
        if self.loss_type == "nll":
            loss = self._nll_loss(F_pred, y, variance)
        elif self.loss_type == "mse":
            loss = self._mse_loss(F_pred, y)
        elif self.loss_type == "hybrid":
            mse = self._mse_loss(F_pred, y)
            nll = self._nll_loss(F_pred, y, variance)
            loss = self.mse_weight * mse + self.nll_weight * nll
        elif self.loss_type == "js":
            loss = energy_landscape_loss(
                E, S, y, T[0], self.kb, self.gamma, loss_type="js"
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Add regularization
        if self.convexity_lambda > 0:
            loss = loss + convexity_penalty_from_ES(
                E, S, X, T, self.kb, self.gamma, self.convexity_lambda
            )

        if self.smoothness_lambda > 0:
            from pycse.sklearn.zenn.losses.constraints import smoothness_penalty

            loss = loss + smoothness_penalty(E, S, X, self.smoothness_lambda)

        return loss

    def _predict_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """Internal prediction."""
        T = jnp.full((X.shape[0],), self.T_default)
        outputs = self.model_.apply(self.params_, X, T)
        return compute_total_helmholtz_energy(
            outputs["E"], outputs["S"], T, self.kb, self.gamma
        )

    def predict(self, X, T=None) -> np.ndarray:
        """
        Predict Helmholtz energy at given points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        T : float or array-like, optional
            Temperature(s). If None, uses T_default.

        Returns
        -------
        F : ndarray of shape (n_samples,)
            Predicted Helmholtz energy.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = jnp.array(X, dtype=jnp.float32)

        if T is None:
            T = self.T_default
        if isinstance(T, (int, float)):
            T = jnp.full((X.shape[0],), float(T))
        else:
            T = jnp.array(T, dtype=jnp.float32)

        outputs = self.model_.apply(self.params_, X, T)
        F = compute_total_helmholtz_energy(
            outputs["E"], outputs["S"], T, self.kb, self.gamma
        )
        return np.array(F)

    def predict_at_temperatures(
        self,
        X,
        temperatures: np.ndarray,
    ) -> np.ndarray:
        """
        Predict Helmholtz energy at multiple temperatures.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        temperatures : array-like of shape (n_temps,)
            Temperature values.

        Returns
        -------
        F : ndarray of shape (n_temps, n_samples)
            Predicted Helmholtz energy at each temperature.
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)
        temperatures = jnp.array(temperatures)

        def predict_at_T(T):
            return self.predict(X, T)

        F = jax.vmap(predict_at_T)(temperatures)
        return np.array(F)

    def compute_derivatives(
        self,
        X,
        T: Optional[float] = None,
        order: int = 1,
    ) -> np.ndarray:
        """
        Compute derivatives of Helmholtz energy via automatic differentiation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        T : float, optional
            Temperature.

        order : int, default=1
            Derivative order (1 for gradient, 2 for Hessian).

        Returns
        -------
        derivatives : ndarray
            For order=1: shape (n_samples, n_features)
            For order=2: shape (n_samples, n_features, n_features)
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        if T is None:
            T = self.T_default

        def F_func(x):
            x = x.reshape(1, -1)
            T_arr = jnp.array([T])
            outputs = self.model_.apply(self.params_, x, T_arr)
            return compute_total_helmholtz_energy(
                outputs["E"], outputs["S"], T_arr, self.kb, self.gamma
            )[0]

        if order == 1:
            grad_fn = jax.vmap(jax.grad(F_func))
            return np.array(grad_fn(X))
        elif order == 2:
            hess_fn = jax.vmap(jax.hessian(F_func))
            return np.array(hess_fn(X))
        else:
            raise ValueError(f"order must be 1 or 2, got {order}")

    def find_equilibrium(
        self,
        x_init,
        T: Optional[float] = None,
        method: str = "newton",
    ) -> np.ndarray:
        """
        Find equilibrium point where dF/dx = 0.

        Parameters
        ----------
        x_init : array-like
            Initial guess.

        T : float, optional
            Temperature.

        method : str, default='newton'
            Optimization method.

        Returns
        -------
        x_eq : ndarray
            Equilibrium coordinates.
        """
        check_is_fitted(self)
        x_init = jnp.array(x_init).flatten()

        if T is None:
            T = self.T_default

        def F_func(x):
            x = x.reshape(1, -1)
            T_arr = jnp.array([T])
            outputs = self.model_.apply(self.params_, x, T_arr)
            return compute_total_helmholtz_energy(
                outputs["E"], outputs["S"], T_arr, self.kb, self.gamma
            )[0]

        # Use gradient descent or Newton's method
        from jax.scipy.optimize import minimize

        result = minimize(F_func, x_init, method="BFGS")
        return np.array(result.x)

    def score(self, X, y, T=None) -> float:
        """
        Return R^2 score on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True Helmholtz energy values.

        T : float or array-like, optional
            Temperature.

        Returns
        -------
        score : float
            R^2 score.
        """
        y_pred = self.predict(X, T)
        y_true = np.asarray(y).flatten()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1 - ss_res / ss_tot

    def predict_with_uncertainty(
        self,
        X,
        T: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict Helmholtz energy with uncertainty estimates.

        Returns the predicted energy along with epistemic and aleatoric
        uncertainty measures derived from ZENN's thermodynamic framework.

        **When using NLL loss (default)**, the aleatoric uncertainty is
        calibrated to match actual noise levels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        T : float, optional
            Temperature.

        Returns
        -------
        dict with:
            - 'prediction': Predicted Helmholtz energy
            - 'epistemic': Configuration disagreement (model uncertainty)
            - 'aleatoric': Noise std (calibrated with NLL loss)
            - 'variance': Noise variance (aleatoric²)
            - 'total': Combined uncertainty
            - 'config_probs': Configuration probabilities
            - 'E_per_config': Energy per configuration
            - 'S_per_config': Entropy per configuration
        """
        from pycse.sklearn.zenn.analysis.uncertainty import (
            epistemic_uncertainty,
            aleatoric_uncertainty,
        )

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

        # Compute total energy prediction
        F_pred = compute_total_helmholtz_energy(E, S, T_arr, self.kb, self.gamma)

        # Uncertainty decomposition
        epist = epistemic_uncertainty(E, p)

        # Variance from S networks (calibrated with NLL loss)
        variance = self._compute_variance(S, p)
        aleat = jnp.sqrt(variance)  # Convert to std

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

    def get_epistemic_uncertainty(
        self,
        X,
        T: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get epistemic uncertainty (configuration disagreement).

        Measures how much the configurations disagree about the energy.
        High epistemic uncertainty suggests model uncertainty that could
        be reduced with more training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        T : float, optional
            Temperature.

        Returns
        -------
        uncertainty : ndarray of shape (n_samples,)
            Epistemic uncertainty for each point.
        """
        from pycse.sklearn.zenn.analysis.uncertainty import epistemic_uncertainty

        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        if T is None:
            T = self.T_default

        T_arr = jnp.full((X.shape[0],), float(T))
        outputs = self.model_.apply(self.params_, X, T_arr)
        p, _ = compute_configuration_probabilities(
            outputs['E'], outputs['S'], T_arr, self.kb, self.gamma
        )

        U = epistemic_uncertainty(outputs['E'], p)
        return np.array(U)

    def get_aleatoric_uncertainty(
        self,
        X,
        T: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get aleatoric uncertainty (inherent data noise).

        When trained with NLL loss (default), this returns calibrated
        noise standard deviation estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        T : float, optional
            Temperature.

        Returns
        -------
        uncertainty : ndarray of shape (n_samples,)
            Aleatoric uncertainty (std) for each point.
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        if T is None:
            T = self.T_default

        T_arr = jnp.full((X.shape[0],), float(T))
        outputs = self.model_.apply(self.params_, X, T_arr)
        p, _ = compute_configuration_probabilities(
            outputs['E'], outputs['S'], T_arr, self.kb, self.gamma
        )

        variance = self._compute_variance(outputs['S'], p)
        return np.array(jnp.sqrt(variance))

    def get_uncertainty_decomposition(
        self,
        X,
        T: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get full uncertainty decomposition into epistemic and aleatoric.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        T : float, optional
            Temperature.

        Returns
        -------
        dict with:
            - 'epistemic': Configuration disagreement in energy
            - 'aleatoric': Noise std from S networks (calibrated with NLL)
            - 'total': Combined uncertainty
        """
        result = self.predict_with_uncertainty(X, T)
        return {
            'epistemic': result['epistemic'],
            'aleatoric': result['aleatoric'],
            'total': result['total'],
        }

    def get_prediction_intervals(
        self,
        X,
        T: Optional[float] = None,
        confidence: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """
        Get prediction intervals based on uncertainty.

        **When trained with NLL loss (default)**, these intervals have
        correct coverage (e.g., 95% intervals contain ~95% of points).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        T : float, optional
            Temperature.

        confidence : float, default=0.95
            Confidence level for intervals.

        Returns
        -------
        dict with:
            - 'mean': Predicted energy (mean)
            - 'lower': Lower bound of interval
            - 'upper': Upper bound of interval
            - 'width': Width of interval
        """
        from scipy import stats

        result = self.predict_with_uncertainty(X, T)

        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence) / 2)

        mean = result['prediction']
        std = result['aleatoric']  # Calibrated with NLL

        lower = mean - z * std
        upper = mean + z * std

        return {
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'width': upper - lower,
        }

    def get_calibration_metrics(
        self,
        X,
        y,
        T: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute calibration metrics to verify uncertainty quality.

        Use this to check if aleatoric uncertainty matches actual noise.
        Ideal calibration_ratio = 1.0, coverage_95 ≈ 0.95.

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
            - 'learned_std': Mean learned aleatoric
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
            'empirical_std': float(empirical_std),
            'learned_std': float(learned_std),
            'calibration_ratio': float(learned_std / empirical_std) if empirical_std > 0 else float('inf'),
            'coverage_95': float(coverage),
        }

    def compute_derivatives_with_uncertainty(
        self,
        X,
        T: Optional[float] = None,
        order: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Compute derivatives with uncertainty estimates.

        Uses the configuration spread to estimate derivative uncertainty.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input coordinates.

        T : float, optional
            Temperature.

        order : int, default=1
            Derivative order (1 for gradient).

        Returns
        -------
        dict with:
            - 'derivatives': Mean derivatives across configurations
            - 'uncertainty': Standard deviation of derivatives
            - 'derivatives_per_config': Derivatives for each configuration
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        if T is None:
            T = self.T_default

        # Compute gradient for each configuration's F(k)
        def F_k_func(x, k):
            x = x.reshape(1, -1)
            T_arr = jnp.array([T])
            outputs = self.model_.apply(self.params_, x, T_arr)
            E_k = outputs['E'][0, k]
            S_k = outputs['S'][0, k]
            return E_k - T * S_k  # F(k) = E(k) - T*S(k)

        if order == 1:
            # Gradient for each configuration
            def grad_per_config(x, k):
                return jax.grad(lambda xi: F_k_func(xi, k))(x)

            # Get gradients for all configurations
            n_configs = self.n_configs
            all_grads = []
            for k in range(n_configs):
                grad_fn = jax.vmap(lambda x: grad_per_config(x, k))
                grads_k = grad_fn(X)
                all_grads.append(grads_k)

            # Stack: (n_samples, n_configs, n_features)
            all_grads = jnp.stack(all_grads, axis=1)

            # Get configuration weights
            T_arr = jnp.full((X.shape[0],), float(T))
            outputs = self.model_.apply(self.params_, X, T_arr)
            p, _ = compute_configuration_probabilities(
                outputs['E'], outputs['S'], T_arr, self.kb, self.gamma
            )

            # Weighted mean gradient
            mean_grad = jnp.sum(
                all_grads * p[:, :, None], axis=1
            )

            # Weighted std of gradients (uncertainty)
            diff = all_grads - mean_grad[:, None, :]
            variance = jnp.sum(p[:, :, None] * diff ** 2, axis=1)
            uncertainty = jnp.sqrt(variance + 1e-10)

            return {
                'derivatives': np.array(mean_grad),
                'uncertainty': np.array(uncertainty),
                'derivatives_per_config': np.array(all_grads),
            }
        else:
            raise ValueError("Currently only order=1 is supported for uncertainty")
