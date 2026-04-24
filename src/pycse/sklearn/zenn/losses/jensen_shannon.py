"""
Jensen-Shannon divergence loss for energy landscape reconstruction (Eq. 17-18).

This loss is used for regression tasks where we want to match a probability
distribution derived from the energy landscape to experimental/simulated data.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Callable

from pycse.sklearn.zenn.utils.thermodynamics import (
    compute_total_helmholtz_energy,
    compute_configuration_probabilities,
)


def kl_divergence(
    p: jnp.ndarray,
    q: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    Compute KL divergence KL(P || Q) = sum P * log(P/Q).

    Args:
        p: First distribution (reference)
        q: Second distribution (approximation)
        eps: Small constant for numerical stability

    Returns:
        KL divergence value
    """
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    return jnp.sum(p * jnp.log(p / q))


def jensen_shannon_divergence(
    p: jnp.ndarray,
    q: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    Compute Jensen-Shannon divergence (Eq. 17).

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = (P + Q) / 2

    Args:
        p: First distribution (e.g., experimental data)
        q: Second distribution (e.g., model prediction)
        eps: Small constant for numerical stability

    Returns:
        JS divergence (scalar)
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def energy_based_probability(
    F: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
) -> jnp.ndarray:
    """
    Compute Boltzmann probability from Helmholtz energy.

    p(x) = exp(-F(x)/(kb*T)) / Z

    Args:
        F: Helmholtz energy values at each point
        T: Temperature
        kb: Boltzmann constant

    Returns:
        Normalized probability distribution
    """
    log_unnorm = -F / (kb * T)
    log_Z = jax.scipy.special.logsumexp(log_unnorm)
    log_p = log_unnorm - log_Z
    return jnp.exp(log_p)


def jensen_shannon_loss(
    E: jnp.ndarray,
    S: jnp.ndarray,
    y_true: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    Jensen-Shannon divergence loss for energy landscape fitting.

    This computes the JS divergence between the target distribution y_true
    and the model's predicted distribution based on zentropy theory.

    Args:
        E: Energy predictions from model, shape (n_points, n_configs)
        S: Entropy predictions from model, shape (n_points, n_configs)
        y_true: Target probability distribution, shape (n_points,)
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        eps: Numerical stability constant

    Returns:
        JS divergence loss (scalar)
    """
    # Compute total Helmholtz energy at each point
    F_total = compute_total_helmholtz_energy(E, S, T, kb, gamma)

    # Convert to probability distribution
    p_pred = energy_based_probability(F_total, T, kb)

    # Normalize target distribution
    y_true = y_true / jnp.sum(y_true)

    return jensen_shannon_divergence(y_true, p_pred, eps)


def mse_loss(
    y_pred: jnp.ndarray,
    y_true: jnp.ndarray,
    reduction: str = "mean",
) -> jnp.ndarray:
    """
    Mean squared error loss for direct regression.

    Args:
        y_pred: Model predictions
        y_true: Target values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MSE loss value
    """
    squared_error = (y_pred - y_true) ** 2

    if reduction == "mean":
        return jnp.mean(squared_error)
    elif reduction == "sum":
        return jnp.sum(squared_error)
    else:
        return squared_error


def energy_landscape_loss(
    E: jnp.ndarray,
    S: jnp.ndarray,
    F_target: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    loss_type: str = "mse",
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    Combined loss for energy landscape reconstruction.

    Args:
        E: Energy predictions, shape (n_points, n_configs)
        S: Entropy predictions, shape (n_points, n_configs)
        F_target: Target Helmholtz energy values, shape (n_points,)
        T: Temperature (scalar or array)
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        loss_type: 'mse' for direct fitting, 'js' for distribution matching
        eps: Numerical stability constant

    Returns:
        Loss value (scalar)
    """
    # Compute predicted total Helmholtz energy
    F_pred = compute_total_helmholtz_energy(E, S, T, kb, gamma)

    if loss_type == "mse":
        return mse_loss(F_pred, F_target)
    elif loss_type == "js":
        # Convert energies to probabilities for JS divergence
        p_pred = energy_based_probability(F_pred, T, kb)
        p_target = energy_based_probability(F_target, T, kb)
        return jensen_shannon_divergence(p_target, p_pred, eps)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def derivative_loss(
    F_func: Callable,
    x: jnp.ndarray,
    dF_target: jnp.ndarray,
    order: int = 1,
) -> jnp.ndarray:
    """
    Loss on derivatives of the energy landscape.

    This can be used for Sobolev-style training when derivative
    information is available.

    Args:
        F_func: Function that computes F(x) -> scalar
        x: Input points, shape (n_points, n_features)
        dF_target: Target derivatives
        order: Derivative order (1 or 2)

    Returns:
        MSE loss on derivatives
    """
    if order == 1:
        # First derivative (gradient)
        grad_F = jax.vmap(jax.grad(F_func))(x)
        return mse_loss(grad_F, dF_target)
    elif order == 2:
        # Second derivative (Hessian diagonal)
        def hess_diag(xi):
            H = jax.hessian(F_func)(xi)
            return jnp.diag(H)

        hess_F = jax.vmap(hess_diag)(x)
        return mse_loss(hess_F, dF_target)
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")
