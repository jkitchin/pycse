"""
Derivative computation utilities using JAX automatic differentiation.

This module provides tools for computing first and second order derivatives
of the energy landscape, which are crucial for:
- Finding equilibrium points (dF/dx = 0)
- Identifying phase transitions (d²F/dx² = 0)
- Computing thermodynamic properties
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional, Dict, Any
import numpy as np


def compute_gradient(
    F_func: Callable,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the gradient dF/dx at given points.

    Args:
        F_func: Function F(x) -> scalar
        x: Input points, shape (n_samples, n_features) or (n_features,)

    Returns:
        Gradient, same shape as x
    """
    if x.ndim == 1:
        return jax.grad(F_func)(x)
    else:
        return jax.vmap(jax.grad(F_func))(x)


def compute_hessian(
    F_func: Callable,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the Hessian matrix d²F/dxdx at given points.

    Args:
        F_func: Function F(x) -> scalar
        x: Input points, shape (n_samples, n_features) or (n_features,)

    Returns:
        Hessian matrix:
        - If x is 1D: shape (n_features, n_features)
        - If x is 2D: shape (n_samples, n_features, n_features)
    """
    if x.ndim == 1:
        return jax.hessian(F_func)(x)
    else:
        return jax.vmap(jax.hessian(F_func))(x)


def compute_derivatives(
    F_func: Callable,
    x: jnp.ndarray,
    order: int = 1,
) -> jnp.ndarray:
    """
    Compute derivatives of arbitrary order.

    Args:
        F_func: Function F(x) -> scalar
        x: Input points
        order: Derivative order (1, 2, or 3)

    Returns:
        Derivatives of specified order
    """
    if order == 1:
        return compute_gradient(F_func, x)
    elif order == 2:
        return compute_hessian(F_func, x)
    elif order == 3:
        # Third derivative (tensor)
        def hess_func(xi):
            return jax.hessian(F_func)(xi)

        if x.ndim == 1:
            return jax.jacfwd(hess_func)(x)
        else:
            return jax.vmap(jax.jacfwd(hess_func))(x)
    else:
        raise ValueError(f"order must be 1, 2, or 3, got {order}")


def compute_directional_derivative(
    F_func: Callable,
    x: jnp.ndarray,
    direction: jnp.ndarray,
    order: int = 1,
) -> jnp.ndarray:
    """
    Compute directional derivative along a given direction.

    Args:
        F_func: Function F(x) -> scalar
        x: Point at which to compute derivative
        direction: Direction vector (will be normalized)
        order: Derivative order

    Returns:
        Directional derivative value
    """
    direction = direction / jnp.linalg.norm(direction)

    if order == 1:
        grad = compute_gradient(F_func, x)
        return jnp.dot(grad, direction)
    elif order == 2:
        hess = compute_hessian(F_func, x)
        return jnp.dot(direction, jnp.dot(hess, direction))
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")


def compute_laplacian(
    F_func: Callable,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the Laplacian (trace of Hessian) at given points.

    The Laplacian is useful for detecting local minima/maxima and
    computing certain thermodynamic quantities.

    Args:
        F_func: Function F(x) -> scalar
        x: Input points

    Returns:
        Laplacian values
    """
    hess = compute_hessian(F_func, x)
    if x.ndim == 1:
        return jnp.trace(hess)
    else:
        return jax.vmap(jnp.trace)(hess)


def compute_eigenvalues(
    F_func: Callable,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute eigenvalues of the Hessian matrix.

    Useful for stability analysis:
    - All positive eigenvalues -> local minimum (stable)
    - All negative eigenvalues -> local maximum (unstable)
    - Mixed signs -> saddle point

    Args:
        F_func: Function F(x) -> scalar
        x: Input points

    Returns:
        Eigenvalues, shape (..., n_features)
    """
    hess = compute_hessian(F_func, x)
    if x.ndim == 1:
        return jnp.linalg.eigvalsh(hess)
    else:
        return jax.vmap(jnp.linalg.eigvalsh)(hess)


def find_zero_crossings(
    values: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Find approximate locations where values cross zero.

    Useful for finding equilibrium points (gradient = 0) or
    inflection points (second derivative = 0).

    Args:
        values: Function values at grid points
        x: Corresponding x coordinates

    Returns:
        Approximate x locations of zero crossings
    """
    # Find sign changes
    signs = jnp.sign(values)
    sign_changes = jnp.diff(signs) != 0

    # Linear interpolation for zero crossing locations
    zero_crossings = []
    for i in range(len(sign_changes)):
        if sign_changes[i]:
            # Linear interpolation
            x0, x1 = x[i], x[i + 1]
            v0, v1 = values[i], values[i + 1]
            x_zero = x0 - v0 * (x1 - x0) / (v1 - v0)
            zero_crossings.append(x_zero)

    return jnp.array(zero_crossings) if zero_crossings else jnp.array([])


def create_F_function_from_model(
    model,
    params: Dict[str, Any],
    T: float,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> Callable:
    """
    Create a scalar function F(x) from a trained ZENN model.

    This wraps the model to provide a simple F(x) interface
    suitable for derivative computations.

    Args:
        model: ZENN model instance
        params: Model parameters
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale

    Returns:
        Function F: x -> scalar Helmholtz energy
    """
    from pycse.sklearn.zenn.utils.thermodynamics import compute_total_helmholtz_energy

    def F_func(x):
        x = x.reshape(1, -1)
        T_arr = jnp.array([T])
        outputs = model.apply(params, x, T_arr)
        return compute_total_helmholtz_energy(
            outputs["E"], outputs["S"], T_arr, kb, gamma
        )[0]

    return F_func


def compute_derivatives_on_grid(
    F_func: Callable,
    x_grid: jnp.ndarray,
    orders: Tuple[int, ...] = (1, 2),
) -> Dict[int, jnp.ndarray]:
    """
    Compute multiple derivative orders on a grid.

    Args:
        F_func: Function F(x) -> scalar
        x_grid: Grid of x values, shape (n_points,) or (n_points, n_features)
        orders: Tuple of derivative orders to compute

    Returns:
        Dictionary mapping order -> derivative values
    """
    results = {}
    for order in orders:
        results[order] = compute_derivatives(F_func, x_grid, order)
    return results


def numerical_derivative(
    F_func: Callable,
    x: jnp.ndarray,
    order: int = 1,
    h: float = 1e-5,
) -> jnp.ndarray:
    """
    Compute numerical derivative using finite differences.

    Useful for validating automatic differentiation results.

    Args:
        F_func: Function F(x) -> scalar
        x: Point at which to compute derivative
        order: Derivative order (1 or 2)
        h: Step size for finite differences

    Returns:
        Numerical derivative
    """
    n = x.shape[0] if x.ndim > 0 else 1

    if order == 1:
        # Central difference: (F(x+h) - F(x-h)) / (2h)
        grad = jnp.zeros(n)
        for i in range(n):
            x_plus = x.at[i].add(h) if x.ndim > 0 else x + h
            x_minus = x.at[i].add(-h) if x.ndim > 0 else x - h
            grad = grad.at[i].set((F_func(x_plus) - F_func(x_minus)) / (2 * h))
        return grad

    elif order == 2:
        # Second derivative: (F(x+h) - 2*F(x) + F(x-h)) / h²
        hess = jnp.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    x_plus = x.at[i].add(h)
                    x_minus = x.at[i].add(-h)
                    hess = hess.at[i, i].set(
                        (F_func(x_plus) - 2 * F_func(x) + F_func(x_minus)) / (h**2)
                    )
                else:
                    x_pp = x.at[i].add(h).at[j].add(h)
                    x_pm = x.at[i].add(h).at[j].add(-h)
                    x_mp = x.at[i].add(-h).at[j].add(h)
                    x_mm = x.at[i].add(-h).at[j].add(-h)
                    hess = hess.at[i, j].set(
                        (F_func(x_pp) - F_func(x_pm) - F_func(x_mp) + F_func(x_mm))
                        / (4 * h**2)
                    )
        return hess

    else:
        raise ValueError(f"order must be 1 or 2, got {order}")
