"""
Thermodynamics utilities implementing zentropy theory.

This module provides functions for computing thermodynamic quantities
based on zentropy theory (Eqs. 1-7 from the paper).
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def compute_helmholtz_energy(
    E: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Helmholtz energy F^(k) = E^(k) - T * S^(k) for each configuration.

    Args:
        E: Internal energy, shape (batch_size, n_configs) or (n_configs,)
        S: Entropy, shape (batch_size, n_configs) or (n_configs,)
        T: Temperature, scalar or shape (batch_size,)

    Returns:
        Helmholtz energy F, same shape as E and S
    """
    if T.ndim == 0:
        return E - T * S
    elif T.ndim == 1 and E.ndim == 2:
        return E - T[:, None] * S
    else:
        return E - T * S


def compute_configuration_probabilities(
    E: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute configuration probabilities via zentropy theory (Eq. 6).

    p^(k) = (1/Z) * exp(-F^(k)/(kb*T) - (S^(k)/(gamma*kb))^2)

    Args:
        E: Internal energy, shape (batch_size, n_configs)
        S: Entropy, shape (batch_size, n_configs)
        T: Temperature, scalar or shape (batch_size,)
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale parameter

    Returns:
        Tuple of (probabilities, log_probabilities), each shape (batch_size, n_configs)
    """
    F = compute_helmholtz_energy(E, S, T)

    # Handle temperature broadcasting
    if T.ndim == 0:
        T_expanded = T
    else:
        T_expanded = T[:, None] if T.ndim == 1 and E.ndim == 2 else T

    # Compute scores (log-unnormalized probabilities)
    scores = -F / (kb * T_expanded) - (S / (gamma * kb)) ** 2

    # Normalize via logsumexp for numerical stability
    log_Z = jax.scipy.special.logsumexp(scores, axis=-1, keepdims=True)
    log_p = scores - log_Z
    p = jnp.exp(log_p)

    return p, log_p


def compute_partition_function(
    E: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> jnp.ndarray:
    """
    Compute the partition function Z (Eq. 6).

    Z = sum_k exp(-F^(k)/(kb*T) - (S^(k)/(gamma*kb))^2)

    Args:
        E: Internal energy, shape (batch_size, n_configs)
        S: Entropy, shape (batch_size, n_configs)
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale

    Returns:
        Partition function Z, shape (batch_size,)
    """
    F = compute_helmholtz_energy(E, S, T)

    if T.ndim == 0:
        T_expanded = T
    else:
        T_expanded = T[:, None] if T.ndim == 1 and E.ndim == 2 else T

    scores = -F / (kb * T_expanded) - (S / (gamma * kb)) ** 2
    log_Z = jax.scipy.special.logsumexp(scores, axis=-1)
    return jnp.exp(log_Z)


def compute_total_entropy(
    E: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> jnp.ndarray:
    """
    Compute total system entropy via zentropy theory (Eq. 1).

    S_total = sum_k p^(k) * S^(k) - kb * sum_k p^(k) * ln(p^(k))

    The first term is the weighted average of configuration entropies,
    and the second term is the statistical/mixing entropy.

    Args:
        E: Internal energy, shape (batch_size, n_configs)
        S: Entropy per configuration, shape (batch_size, n_configs)
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale

    Returns:
        Total entropy, shape (batch_size,)
    """
    p, log_p = compute_configuration_probabilities(E, S, T, kb, gamma)

    # Configuration entropy contribution
    S_config = jnp.sum(p * S, axis=-1)

    # Statistical/mixing entropy contribution (always positive)
    S_mixing = -kb * jnp.sum(p * log_p, axis=-1)

    return S_config + S_mixing


def compute_total_helmholtz_energy(
    E: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> jnp.ndarray:
    """
    Compute total Helmholtz energy of the system (Eq. 7).

    F_total = sum_k p^(k) * F^(k) + kb*T * sum_k p^(k) * ln(p^(k))

    Args:
        E: Internal energy, shape (batch_size, n_configs)
        S: Entropy, shape (batch_size, n_configs)
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale

    Returns:
        Total Helmholtz energy, shape (batch_size,)
    """
    F = compute_helmholtz_energy(E, S, T)
    p, log_p = compute_configuration_probabilities(E, S, T, kb, gamma)

    if T.ndim == 0:
        T_scalar = T
    else:
        T_scalar = T

    # Weighted average of configuration Helmholtz energies
    F_weighted = jnp.sum(p * F, axis=-1)

    # Entropic contribution from configuration mixing
    F_mixing = kb * T_scalar * jnp.sum(p * log_p, axis=-1)

    return F_weighted + F_mixing


def compute_expected_energy(
    E: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> jnp.ndarray:
    """
    Compute expected internal energy E_total = sum_k p^(k) * E^(k).

    Args:
        E: Internal energy, shape (batch_size, n_configs)
        S: Entropy, shape (batch_size, n_configs)
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale

    Returns:
        Expected energy, shape (batch_size,)
    """
    p, _ = compute_configuration_probabilities(E, S, T, kb, gamma)
    return jnp.sum(p * E, axis=-1)


def compute_entropy_fluctuation(
    E: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> jnp.ndarray:
    """
    Compute entropy fluctuation constraint sum_k p^(k) * (S^(k))^2.

    This is the third constraint in the zentropy optimization (Eq. 2).

    Args:
        E: Internal energy
        S: Entropy
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale

    Returns:
        Entropy fluctuation, shape (batch_size,)
    """
    p, _ = compute_configuration_probabilities(E, S, T, kb, gamma)
    return jnp.sum(p * S**2, axis=-1)
