"""Neural network architectures for ZENN."""

from pycse.sklearn.zenn.networks.mlp import ConfigurationNetwork, create_configuration_networks
from pycse.sklearn.zenn.networks.kan import ChebyKANLayer, ConfigurationNetworkKAN

__all__ = [
    "ConfigurationNetwork",
    "ChebyKANLayer",
    "ConfigurationNetworkKAN",
    "create_configuration_networks",
]
