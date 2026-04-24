"""
JAX/Flax neural network architectures for ZENN.

Each configuration k in ZENN has two networks:
- E_net: Models internal energy E^(k)(x, T)
- S_net: Models intrinsic entropy S^(k)(x, T)
"""

from typing import Sequence, Tuple, Dict, Any
import jax
import jax.numpy as jnp
from flax import linen as nn


class ConfigurationNetwork(nn.Module):
    """
    A shallow MLP that models either E^(k) or S^(k) for a single configuration.

    Architecture follows the paper: 2 hidden layers with tanh activation,
    output layer with no bias (as in the original implementation).

    Attributes:
        hidden_dims: Sequence of hidden layer dimensions, e.g., (8, 8)
        activation: Activation function name ('tanh', 'relu', 'gelu')
    """

    hidden_dims: Sequence[int] = (8, 8)
    activation: str = "tanh"

    def setup(self):
        self.activation_fn = {
            "tanh": nn.tanh,
            "relu": nn.relu,
            "gelu": nn.gelu,
            "silu": nn.silu,
        }.get(self.activation, nn.tanh)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the configuration network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
               For energy landscapes: input_dim = n_features + 1 (including T)
               For classification: input_dim = n_features

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = self.activation_fn(x)
        # Output layer with no bias, following the paper
        return nn.Dense(1, use_bias=False)(x)


class ConfigurationPair(nn.Module):
    """
    A pair of networks (E_net, S_net) for a single configuration.

    This models both E^(k)(x, T) and S^(k)(x, T) for configuration k.
    """

    hidden_dims: Sequence[int] = (8, 8)
    activation: str = "tanh"
    network_type: str = "mlp"
    degree: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute energy and entropy for this configuration.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (E, S), each of shape (batch_size, 1)
        """
        if self.network_type == "kan":
            from pycse.sklearn.zenn.networks.kan import ConfigurationNetworkKAN

            E = ConfigurationNetworkKAN(
                hidden_dims=self.hidden_dims, degree=self.degree, name="E_net"
            )(x)
            S = ConfigurationNetworkKAN(
                hidden_dims=self.hidden_dims, degree=self.degree, name="S_net"
            )(x)
        else:
            E = ConfigurationNetwork(
                hidden_dims=self.hidden_dims, activation=self.activation, name="E_net"
            )(x)
            S = ConfigurationNetwork(
                hidden_dims=self.hidden_dims, activation=self.activation, name="S_net"
            )(x)
        # Ensure entropy is non-negative (S >= 0 by thermodynamics)
        S = nn.softplus(S)
        return E, S


class ZENNModel(nn.Module):
    """
    Full ZENN model with K configuration pairs.

    This is the main neural network architecture that implements zentropy theory.
    Each of K configurations has its own (E_net, S_net) pair.

    Attributes:
        n_configs: Number of configurations K
        hidden_dims: Hidden layer dimensions for each network
        activation: Activation function name
        kb: Boltzmann constant (default 1.0)
        gamma: Entropy fluctuation scale parameter
        network_type: Backbone type ('mlp' or 'kan')
        degree: Chebyshev polynomial degree (only used when network_type='kan')
    """

    n_configs: int = 6
    hidden_dims: Sequence[int] = (8, 8)
    activation: str = "tanh"
    kb: float = 1.0
    gamma: float = 100.0
    network_type: str = "mlp"
    degree: int = 3

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, T: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through the full ZENN model.

        Args:
            x: Input features of shape (batch_size, n_features)
            T: Temperature(s) - scalar or shape (batch_size,) or (n_temps,)

        Returns:
            Dictionary containing:
                - 'E': Energy for each config, shape (batch_size, n_configs)
                - 'S': Entropy for each config, shape (batch_size, n_configs)
                - 'F': Helmholtz energy for each config
                - 'p': Configuration probabilities
                - 'F_total': Total Helmholtz energy
        """
        batch_size = x.shape[0]

        # Handle temperature broadcasting
        if T.ndim == 0:
            T = jnp.full((batch_size,), T)
        elif T.shape[0] != batch_size:
            # Broadcast temperature to batch size
            T = jnp.broadcast_to(T, (batch_size,))

        # Concatenate features with temperature for input
        # Shape: (batch_size, n_features + 1)
        x_T = jnp.concatenate([x, T[:, None]], axis=-1)

        # Compute E and S for each configuration
        E_list = []
        S_list = []
        for k in range(self.n_configs):
            config_net = ConfigurationPair(
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                network_type=self.network_type,
                degree=self.degree,
                name=f"config_{k}",
            )
            E_k, S_k = config_net(x_T)
            E_list.append(E_k)
            S_list.append(S_k)

        # Stack: (batch_size, n_configs)
        E = jnp.concatenate(E_list, axis=-1)
        S = jnp.concatenate(S_list, axis=-1)

        # Compute Helmholtz energy F^(k) = E^(k) - T * S^(k)
        F = E - T[:, None] * S

        # Compute configuration probabilities via Eq. 6
        # p^(k) = (1/Z) * exp(-F^(k)/(kb*T) - (S^(k)/(gamma*kb))^2)
        scores = -F / (self.kb * T[:, None]) - (S / (self.gamma * self.kb)) ** 2
        log_Z = jax.scipy.special.logsumexp(scores, axis=-1, keepdims=True)
        log_p = scores - log_Z
        p = jnp.exp(log_p)

        # Compute total Helmholtz energy via Eq. 7
        # F_total = sum_k p^(k) * F^(k) + kb*T * sum_k p^(k) * ln(p^(k))
        F_total = jnp.sum(p * F, axis=-1) + self.kb * T * jnp.sum(
            p * log_p, axis=-1
        )

        return {
            "E": E,
            "S": S,
            "F": F,
            "p": p,
            "log_p": log_p,
            "F_total": F_total,
            "T": T,
        }


def create_configuration_networks(
    n_configs: int = 6,
    hidden_dims: Sequence[int] = (8, 8),
    activation: str = "tanh",
    kb: float = 1.0,
    gamma: float = 100.0,
    network_type: str = "mlp",
    degree: int = 3,
) -> ZENNModel:
    """
    Factory function to create a ZENN model.

    Args:
        n_configs: Number of configurations K
        hidden_dims: Hidden layer dimensions
        activation: Activation function name
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        network_type: Backbone type ('mlp' or 'kan')
        degree: Chebyshev polynomial degree (only used when network_type='kan')

    Returns:
        ZENNModel instance
    """
    return ZENNModel(
        n_configs=n_configs,
        hidden_dims=hidden_dims,
        activation=activation,
        kb=kb,
        gamma=gamma,
        network_type=network_type,
        degree=degree,
    )


def init_zenn_params(
    model: ZENNModel,
    input_dim: int,
    key: jax.random.PRNGKey,
) -> Dict[str, Any]:
    """
    Initialize ZENN model parameters.

    Args:
        model: ZENNModel instance
        input_dim: Number of input features (excluding temperature)
        key: JAX random key

    Returns:
        Initialized parameter dictionary
    """
    # Dummy input for initialization
    dummy_x = jnp.zeros((1, input_dim))
    dummy_T = jnp.ones((1,))
    params = model.init(key, dummy_x, dummy_T)
    return params
