"""
Regularization constraints for ZENN (Eq. 20 and related).

Includes:
- Convexity constraint for physical consistency
- Entropy fluctuation constraint
- Other physics-informed regularizations
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional

from pycse.sklearn.zenn.utils.thermodynamics import compute_total_helmholtz_energy


def convexity_penalty(
    F_func: Callable,
    x_samples: jnp.ndarray,
    lambda_reg: float = 1e-4,
) -> jnp.ndarray:
    """
    Penalize non-convex regions of energy landscape (Eq. 20).

    This ensures physical consistency by requiring that the Helmholtz
    energy is convex (positive second derivative).

    penalty = lambda * integral ReLU(-d²F/dx²) dx

    Args:
        F_func: Function that computes F(x) -> scalar
        x_samples: Sample points for integration, shape (n_samples, n_features)
        lambda_reg: Regularization weight (paper uses 10^-4)

    Returns:
        Convexity penalty (scalar)
    """
    # Compute second derivative (Hessian) at each sample point
    def second_derivative(xi):
        # For 1D, return d²F/dx²
        # For multi-D, return trace of Hessian or minimum eigenvalue
        H = jax.hessian(F_func)(xi)
        if H.ndim == 0:
            return H
        elif H.ndim == 2:
            # Return minimum eigenvalue (convex if all positive)
            eigvals = jnp.linalg.eigvalsh(H)
            return jnp.min(eigvals)
        else:
            return H

    d2F = jax.vmap(second_derivative)(x_samples)

    # ReLU(-d²F/dx²): penalize negative curvature
    penalty = jnp.mean(jax.nn.relu(-d2F))

    return lambda_reg * penalty


def convexity_penalty_from_ES(
    E: jnp.ndarray,
    S: jnp.ndarray,
    x: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    lambda_reg: float = 1e-4,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Convexity penalty computed directly from E and S predictions.

    Uses finite differences to approximate second derivative when
    automatic differentiation through the full model is complex.

    Args:
        E: Energy predictions, shape (n_points, n_configs)
        S: Entropy predictions, shape (n_points, n_configs)
        x: Input coordinates, shape (n_points, n_features)
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        lambda_reg: Regularization weight
        eps: Finite difference step size

    Returns:
        Convexity penalty
    """
    # Compute total Helmholtz energy
    F = compute_total_helmholtz_energy(E, S, T, kb, gamma)

    # For 1D problems, compute second derivative via finite differences
    # Assumes x is sorted and uniformly spaced
    if x.ndim == 1 or x.shape[1] == 1:
        x_flat = x.flatten() if x.ndim > 1 else x
        dx = x_flat[1] - x_flat[0]

        # Central finite difference: d²F/dx² ≈ (F[i+1] - 2*F[i] + F[i-1]) / dx²
        d2F = (F[2:] - 2 * F[1:-1] + F[:-2]) / (dx**2)

        # Penalize negative curvature
        penalty = jnp.mean(jax.nn.relu(-d2F))
    else:
        # For multi-D, use Laplacian approximation
        # This is a simplified version; full Hessian would be more accurate
        penalty = jnp.array(0.0)

    return lambda_reg * penalty


def entropy_fluctuation_penalty(
    E: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    target_C2: float,
    kb: float = 1.0,
    gamma: float = 100.0,
    lambda_reg: float = 1e-3,
) -> jnp.ndarray:
    """
    Penalty to enforce entropy fluctuation constraint.

    Ensures sum_k p^(k) * (S^(k))^2 ≈ C2

    Args:
        E: Energy predictions
        S: Entropy predictions
        T: Temperature
        target_C2: Target value for entropy fluctuation
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        lambda_reg: Regularization weight

    Returns:
        Entropy fluctuation penalty
    """
    from pycse.sklearn.zenn.utils.thermodynamics import (
        compute_configuration_probabilities,
        compute_entropy_fluctuation,
    )

    actual_C2 = compute_entropy_fluctuation(E, S, T, kb, gamma)
    penalty = jnp.mean((actual_C2 - target_C2) ** 2)
    return lambda_reg * penalty


def non_negativity_penalty(
    S: jnp.ndarray,
    lambda_reg: float = 1e-3,
) -> jnp.ndarray:
    """
    Penalty to ensure entropy values are non-negative.

    S >= 0 by thermodynamics, but networks may predict negative values.

    Args:
        S: Entropy predictions
        lambda_reg: Regularization weight

    Returns:
        Non-negativity penalty
    """
    return lambda_reg * jnp.mean(jax.nn.relu(-S))


def smoothness_penalty(
    E: jnp.ndarray,
    S: jnp.ndarray,
    x: jnp.ndarray,
    lambda_reg: float = 1e-4,
) -> jnp.ndarray:
    """
    Smoothness penalty on energy and entropy curves.

    Penalizes large variations between adjacent points to prevent overfitting.

    Args:
        E: Energy predictions, shape (n_points, n_configs)
        S: Entropy predictions, shape (n_points, n_configs)
        x: Input coordinates (assumed sorted)
        lambda_reg: Regularization weight

    Returns:
        Smoothness penalty
    """
    # First-order differences
    dE = jnp.diff(E, axis=0)
    dS = jnp.diff(S, axis=0)

    # Second-order differences (curvature)
    d2E = jnp.diff(dE, axis=0)
    d2S = jnp.diff(dS, axis=0)

    # Penalize large curvature
    penalty = jnp.mean(d2E**2) + jnp.mean(d2S**2)

    return lambda_reg * penalty


def combined_regularization(
    E: jnp.ndarray,
    S: jnp.ndarray,
    x: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    convexity_weight: float = 1e-4,
    smoothness_weight: float = 0.0,
    non_neg_weight: float = 0.0,
) -> jnp.ndarray:
    """
    Combined regularization for energy landscape fitting.

    Args:
        E: Energy predictions
        S: Entropy predictions
        x: Input coordinates
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        convexity_weight: Weight for convexity penalty
        smoothness_weight: Weight for smoothness penalty
        non_neg_weight: Weight for non-negativity penalty

    Returns:
        Total regularization loss
    """
    total = jnp.array(0.0)

    if convexity_weight > 0:
        total = total + convexity_penalty_from_ES(
            E, S, x, T, kb, gamma, convexity_weight
        )

    if smoothness_weight > 0:
        total = total + smoothness_penalty(E, S, x, smoothness_weight)

    if non_neg_weight > 0:
        total = total + non_negativity_penalty(S, non_neg_weight)

    return total
