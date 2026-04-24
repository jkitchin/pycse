"""
Base estimator class for ZENN.

Provides common functionality for both classification and regression,
following sklearn's estimator API.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Sequence, Union

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from pycse.sklearn.zenn.networks.mlp import ZENNModel, init_zenn_params
from pycse.sklearn.zenn.temperature.learnable import LearnableTemperatureSet


class ZENNBase(BaseEstimator, ABC):
    """
    Base class for ZENN estimators.

    This provides the common infrastructure for both ZENNClassifier and
    ZENNRegressor, including parameter initialization, training loop,
    and temperature learning.

    Parameters
    ----------
    n_configs : int, default=6
        Number of configurations K in the zentropy model.
        More configurations can capture more complex energy landscapes.

    hidden_dims : tuple of int, default=(8, 8)
        Dimensions of hidden layers in each configuration network.

    n_temperatures : int, default=4
        Number of learnable temperature modes for multi-source data.
        Set to 1 for homogeneous data.

    kb : float, default=1.0
        Boltzmann constant. Can be adjusted for scaling.

    gamma : float, default=100.0
        Entropy fluctuation scale parameter.
        Controls the influence of entropy variance constraint.

    learning_rate : float, default=0.01
        Learning rate for model parameters.

    temperature_lr : float, default=1e-3
        Learning rate for temperature parameters.

    max_epochs : int, default=1000
        Maximum number of training epochs.

    batch_size : int, default=32
        Mini-batch size for training.

    early_stopping : bool, default=False
        Whether to use early stopping based on validation loss.

    patience : int, default=10
        Number of epochs to wait for improvement before stopping.

    validation_fraction : float, default=0.1
        Fraction of training data to use for validation.

    convexity_lambda : float, default=0.0
        Weight for convexity constraint. Set > 0 for energy landscape tasks.

    omega_train : float, default=5.0
        Temperature posterior sharpness during training.

    omega_test : float, default=1.0
        Temperature posterior sharpness during inference.

    activation : str, default='tanh'
        Activation function: 'tanh', 'relu', 'gelu', 'silu'.

    network_type : str, default='mlp'
        Backbone network type: 'mlp' or 'kan' (ChebyKAN).

    degree : int, default=3
        Chebyshev polynomial degree (only used when network_type='kan').

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    model_ : ZENNModel
        The trained JAX/Flax model.

    params_ : dict
        Trained model parameters.

    temperatures_ : ndarray
        Learned temperature values.

    history_ : dict
        Training history with losses.

    n_features_in_ : int
        Number of input features.
    """

    def __init__(
        self,
        n_configs: int = 6,
        hidden_dims: Tuple[int, ...] = (8, 8),
        n_temperatures: int = 4,
        kb: float = 1.0,
        gamma: float = 100.0,
        learning_rate: float = 0.01,
        temperature_lr: float = 1e-3,
        max_epochs: int = 1000,
        batch_size: int = 32,
        early_stopping: bool = False,
        patience: int = 10,
        validation_fraction: float = 0.1,
        convexity_lambda: float = 0.0,
        omega_train: float = 5.0,
        omega_test: float = 1.0,
        activation: str = "tanh",
        network_type: str = "mlp",
        degree: int = 3,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_configs = n_configs
        self.hidden_dims = hidden_dims
        self.n_temperatures = n_temperatures
        self.kb = kb
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.temperature_lr = temperature_lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.convexity_lambda = convexity_lambda
        self.omega_train = omega_train
        self.omega_test = omega_test
        self.activation = activation
        self.network_type = network_type
        self.degree = degree
        self.random_state = random_state
        self.verbose = verbose

    def _initialize_model(self, n_features: int, n_outputs: int):
        """Initialize the ZENN model and parameters."""
        # Set random key
        if self.random_state is not None:
            key = jax.random.PRNGKey(self.random_state)
        else:
            key = jax.random.PRNGKey(0)

        # Create model
        self.model_ = ZENNModel(
            n_configs=n_outputs if hasattr(self, "_is_classifier") else self.n_configs,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            kb=self.kb,
            gamma=self.gamma,
            network_type=self.network_type,
            degree=self.degree,
        )

        # Initialize parameters
        key, subkey = jax.random.split(key)
        self.params_ = init_zenn_params(self.model_, n_features, subkey)

        # Initialize temperatures
        key, subkey = jax.random.split(key)
        self.temperatures_ = jnp.linspace(0.5, 2.0, self.n_temperatures)

        # Initialize optimizer
        self.optimizer_ = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(self.learning_rate, weight_decay=0.01),
        )
        self.opt_state_ = self.optimizer_.init(self.params_)

        # Temperature optimizer
        if self.n_temperatures > 1:
            self.temp_optimizer_ = optax.adam(self.temperature_lr)
            raw_temps = jnp.zeros(self.n_temperatures)
            self.temp_opt_state_ = self.temp_optimizer_.init(raw_temps)
            self.raw_temps_ = raw_temps

        self._key = key

    def _get_batches(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True):
        """Generate mini-batches for training."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            self._key, subkey = jax.random.split(self._key)
            indices = jax.random.permutation(subkey, indices)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

    @abstractmethod
    def _compute_loss(
        self,
        params: Dict[str, Any],
        X: jnp.ndarray,
        y: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the loss function. To be implemented by subclasses."""
        pass

    @abstractmethod
    def _predict_impl(
        self,
        X: jnp.ndarray,
    ) -> jnp.ndarray:
        """Internal prediction method. To be implemented by subclasses."""
        pass

    def _train_step(
        self,
        params: Dict[str, Any],
        opt_state,
        X: jnp.ndarray,
        y: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> Tuple[Dict[str, Any], Any, jnp.ndarray]:
        """Single training step."""
        loss, grads = jax.value_and_grad(self._compute_loss)(
            params, X, y, temperatures
        )
        updates, new_opt_state = self.optimizer_.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    def fit(self, X, y):
        """
        Fit the ZENN model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X, y = self._validate_data(X, y)
        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32)

        # Store dimensions
        self.n_features_in_ = X.shape[1]
        n_outputs = self._get_n_outputs(y)

        # Initialize model
        self._initialize_model(self.n_features_in_, n_outputs)

        # JIT compile training step
        train_step_jit = jax.jit(self._train_step)

        # Training loop
        self.history_ = {"loss": [], "val_loss": []}
        best_loss = float("inf")
        patience_counter = 0

        # Split validation set if using early stopping
        if self.early_stopping and self.validation_fraction > 0:
            n_val = int(X.shape[0] * self.validation_fraction)
            X_train, X_val = X[:-n_val], X[-n_val:]
            y_train, y_val = y[:-n_val], y[-n_val:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        for epoch in range(self.max_epochs):
            epoch_losses = []

            for X_batch, y_batch in self._get_batches(X_train, y_train):
                X_batch = jnp.array(X_batch)
                y_batch = jnp.array(y_batch)

                self.params_, self.opt_state_, loss = train_step_jit(
                    self.params_,
                    self.opt_state_,
                    X_batch,
                    y_batch,
                    self.temperatures_,
                )
                epoch_losses.append(float(loss))

            avg_loss = np.mean(epoch_losses)
            self.history_["loss"].append(avg_loss)

            # Validation loss
            if X_val is not None:
                val_loss = float(
                    self._compute_loss(
                        self.params_, X_val, y_val, self.temperatures_
                    )
                )
                self.history_["val_loss"].append(val_loss)

                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self._best_params = self.params_.copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        self.params_ = self._best_params
                        break

            if self.verbose and epoch % max(1, self.max_epochs // 10) == 0:
                msg = f"Epoch {epoch}: loss={avg_loss:.6f}"
                if X_val is not None:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)

        return self

    def _validate_data(self, X, y):
        """Validate and preprocess input data."""
        return check_X_y(X, y, multi_output=True, y_numeric=True)

    @abstractmethod
    def _get_n_outputs(self, y: jnp.ndarray) -> int:
        """Get number of outputs from target array."""
        pass

    def get_energy_landscape(
        self,
        X: np.ndarray,
        T: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get the energy landscape at given points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input points.

        T : float or None
            Temperature. If None, uses the first learned temperature.

        Returns
        -------
        dict
            Dictionary containing:
            - 'E': Energy for each configuration
            - 'S': Entropy for each configuration
            - 'F': Helmholtz energy for each configuration
            - 'F_total': Total Helmholtz energy
            - 'p': Configuration probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        if T is None:
            T = self.temperatures_[0]

        T_arr = jnp.full((X.shape[0],), T)
        outputs = self.model_.apply(self.params_, X, T_arr)

        return {k: np.array(v) for k, v in outputs.items()}

    def get_configuration_probabilities(
        self,
        X: np.ndarray,
        T: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get configuration probabilities p^(k) for each input.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input points.

        T : float or None
            Temperature.

        Returns
        -------
        ndarray of shape (n_samples, n_configs)
            Configuration probabilities.
        """
        landscape = self.get_energy_landscape(X, T)
        return landscape["p"]

    def get_temperatures(self) -> np.ndarray:
        """Return the learned temperature values."""
        check_is_fitted(self)
        return np.array(self.temperatures_)

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ("optimizer_", "opt_state_", "temp_optimizer_", "temp_opt_state_"):
            state.pop(k, None)
        return state
