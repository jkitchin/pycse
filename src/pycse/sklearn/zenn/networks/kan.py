"""
ChebyKAN (Chebyshev Kolmogorov-Arnold Network) layers for ZENN.

Provides a KAN backbone using Chebyshev polynomial basis functions as
learnable activations. Drop-in replacement for MLP configuration networks.
"""

from typing import Sequence
import jax.numpy as jnp
from flax import linen as nn


class ChebyKANLayer(nn.Module):
    """
    A single Chebyshev KAN layer.

    Uses Chebyshev polynomials of the first kind as basis functions for
    learnable activations. Input is normalized via tanh to [-1, 1], then
    Chebyshev polynomials T_0..T_d are computed via the recurrence relation
    and contracted with learnable coefficients to produce the output.

    Attributes
    ----------
    out_features : int
        Number of output features.
    degree : int
        Maximum degree of Chebyshev polynomials (default 3).
    """

    out_features: int
    degree: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the ChebyKAN layer.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        jnp.ndarray
            Output tensor of shape (batch_size, out_features).
        """
        in_features = x.shape[-1]

        cheby_coeffs = self.param(
            "cheby_coeffs",
            nn.initializers.normal(stddev=1.0 / (in_features * (self.degree + 1))),
            (in_features, self.out_features, self.degree + 1),
        )

        # Normalize input to [-1, 1] via tanh
        x = jnp.tanh(x)  # (batch, in_features)

        # Compute Chebyshev polynomials via recurrence
        # T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
        T0 = jnp.ones_like(x)  # (batch, in_features)
        T1 = x

        cheby_basis = [T0, T1]
        for _ in range(2, self.degree + 1):
            Tn = 2 * x * cheby_basis[-1] - cheby_basis[-2]
            cheby_basis.append(Tn)

        # Stack: (batch, in_features, degree+1)
        cheby_stack = jnp.stack(cheby_basis[: self.degree + 1], axis=-1)

        # Contract with coefficients: (batch, in_features, degree+1) x
        # (in_features, out_features, degree+1) -> (batch, out_features)
        out = jnp.einsum("bid,iod->bo", cheby_stack, cheby_coeffs)
        return out


class ConfigurationNetworkKAN(nn.Module):
    """
    A KAN-based configuration network using Chebyshev polynomial layers.

    Drop-in replacement for ConfigurationNetwork. Chebyshev polynomials
    serve as learnable activations, so no separate activation function
    is needed.

    Attributes
    ----------
    hidden_dims : Sequence[int]
        Sequence of hidden layer dimensions, e.g., (8, 8).
    degree : int
        Maximum degree of Chebyshev polynomials (default 3).
    """

    hidden_dims: Sequence[int] = (8, 8)
    degree: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the KAN configuration network.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        jnp.ndarray
            Output tensor of shape (batch_size, 1).
        """
        for dim in self.hidden_dims:
            x = ChebyKANLayer(out_features=dim, degree=self.degree)(x)
        # Output layer to scalar
        return ChebyKANLayer(out_features=1, degree=self.degree)(x)
