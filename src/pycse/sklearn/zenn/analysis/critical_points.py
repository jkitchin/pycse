"""
Critical point detection for energy landscapes (Eq. 19).

This module provides tools for finding bifurcation points where
the energy landscape undergoes qualitative changes, such as:
- Phase transitions
- Critical temperatures
- Spinodal decomposition points
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional, Dict, Any
import numpy as np

from pycse.sklearn.zenn.analysis.derivatives import compute_gradient, compute_hessian, compute_eigenvalues


def find_critical_point(
    F_func: Callable,
    x_init: jnp.ndarray,
    T_init: float,
    T_bounds: Tuple[float, float] = (0.1, 10.0),
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[jnp.ndarray, float, bool]:
    """
    Find critical point where gradient = 0 and Hessian has zero eigenvalue (Eq. 19).

    B(x*, T*, ξ) = {
        ∇F(x*, T*) = 0,
        ∇²F(x*, T*) ξ = 0,
        ||ξ|| = 1
    }

    Args:
        F_func: Function F(x, T) -> scalar
        x_init: Initial guess for position
        T_init: Initial guess for temperature
        T_bounds: (T_min, T_max) bounds for temperature
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (x_critical, T_critical, converged)
    """
    from jax.scipy.optimize import minimize

    x_init = jnp.atleast_1d(x_init)
    n_dims = x_init.shape[0]

    def residual(state):
        """Residual for Newton's method."""
        x = state[:n_dims]
        T = jnp.clip(state[n_dims], T_bounds[0], T_bounds[1])

        # Create F at this T
        def F_at_T(xi):
            return F_func(xi, T)

        # Gradient should be zero
        grad = jax.grad(F_at_T)(x)

        # Minimum eigenvalue of Hessian should be zero
        hess = jax.hessian(F_at_T)(x)
        min_eigval = jnp.min(jnp.linalg.eigvalsh(hess))

        # Combined residual
        return jnp.sum(grad**2) + min_eigval**2

    # Initial state
    state_init = jnp.concatenate([x_init, jnp.array([T_init])])

    # Minimize residual
    result = minimize(residual, state_init, method="BFGS")

    x_final = result.x[:n_dims]
    T_final = jnp.clip(result.x[n_dims], T_bounds[0], T_bounds[1])
    converged = result.fun < tol

    return x_final, float(T_final), converged


def find_bifurcation_curve(
    F_func: Callable,
    x_range: Tuple[float, float],
    T_range: Tuple[float, float],
    n_x: int = 100,
    n_T: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Find the bifurcation curve where d²F/dx² = 0 in the (x, T) plane.

    This identifies the spinodal curve separating stable from unstable regions.

    Args:
        F_func: Function F(x, T) -> scalar
        x_range: (x_min, x_max)
        T_range: (T_min, T_max)
        n_x: Number of x grid points
        n_T: Number of T grid points

    Returns:
        Tuple of (x_grid, T_grid, d2F_dx2) where d2F_dx2[i,j] = ∂²F/∂x² at (x_grid[i], T_grid[j])
    """
    x_grid = jnp.linspace(x_range[0], x_range[1], n_x)
    T_grid = jnp.linspace(T_range[0], T_range[1], n_T)

    def compute_d2F(x, T):
        def F_at_T(xi):
            return F_func(jnp.array([xi]), T)

        return jax.grad(jax.grad(F_at_T))(x)

    # Vectorize over x and T
    d2F = jax.vmap(
        lambda T: jax.vmap(lambda x: compute_d2F(x, T))(x_grid)
    )(T_grid)

    return x_grid, T_grid, d2F.T  # Shape: (n_x, n_T)


def find_equilibrium_branches(
    F_func: Callable,
    T_range: Tuple[float, float],
    x_init_guesses: jnp.ndarray,
    n_T: int = 100,
) -> Dict[str, jnp.ndarray]:
    """
    Track equilibrium branches (dF/dx = 0) as temperature varies.

    This traces the bifurcation diagram showing stable/metastable/unstable states.

    Args:
        F_func: Function F(x, T) -> scalar
        T_range: (T_min, T_max)
        x_init_guesses: Initial guesses for equilibrium positions
        n_T: Number of temperature points

    Returns:
        Dictionary with:
        - 'T': Temperature values
        - 'x_branches': List of equilibrium branch arrays
        - 'stability': List of stability indicators (eigenvalue signs)
    """
    from jax.scipy.optimize import minimize

    T_values = jnp.linspace(T_range[0], T_range[1], n_T)
    branches = []
    stabilities = []

    for x_init in x_init_guesses:
        branch_x = []
        branch_stable = []

        x_current = x_init

        for T in T_values:
            # Find equilibrium at this T
            def F_at_T(x):
                return F_func(x, T)

            result = minimize(F_at_T, x_current, method="BFGS")
            x_eq = result.x

            # Check stability (sign of second derivative)
            d2F = jax.hessian(F_at_T)(x_eq)
            if x_eq.ndim == 0 or x_eq.shape[0] == 1:
                stable = float(d2F) > 0
            else:
                eigvals = jnp.linalg.eigvalsh(d2F)
                stable = jnp.all(eigvals > 0)

            branch_x.append(float(x_eq[0]) if x_eq.ndim > 0 else float(x_eq))
            branch_stable.append(stable)

            x_current = x_eq

        branches.append(jnp.array(branch_x))
        stabilities.append(jnp.array(branch_stable))

    return {
        "T": T_values,
        "x_branches": branches,
        "stability": stabilities,
    }


def compute_critical_temperature(
    F_func: Callable,
    x_range: Tuple[float, float],
    T_range: Tuple[float, float],
    n_x: int = 100,
    n_T: int = 100,
) -> float:
    """
    Compute the critical temperature where bifurcation occurs.

    The critical point is where both first and second derivatives
    with respect to x are zero simultaneously.

    Args:
        F_func: Function F(x, T) -> scalar
        x_range: Range of x values to search
        T_range: Range of temperatures to search
        n_x: Grid resolution in x
        n_T: Grid resolution in T

    Returns:
        Critical temperature T*
    """
    x_grid, T_grid, d2F = find_bifurcation_curve(F_func, x_range, T_range, n_x, n_T)

    # Also compute first derivative
    def compute_dF(x, T):
        def F_at_T(xi):
            return F_func(jnp.array([xi]), T)

        return jax.grad(F_at_T)(x)

    dF = jax.vmap(
        lambda T: jax.vmap(lambda x: compute_dF(x, T))(x_grid)
    )(T_grid).T

    # Find where both dF ≈ 0 and d2F ≈ 0
    combined = jnp.abs(dF) + jnp.abs(d2F)
    idx = jnp.unravel_index(jnp.argmin(combined), combined.shape)

    return float(T_grid[idx[1]])


def identify_phase_regions(
    F_func: Callable,
    x_range: Tuple[float, float],
    T_range: Tuple[float, float],
    n_x: int = 100,
    n_T: int = 100,
) -> Dict[str, jnp.ndarray]:
    """
    Identify stable, metastable, and unstable regions in (x, T) space.

    Regions are classified by the sign of the second derivative:
    - d²F/dx² > 0: Stable (local minimum)
    - d²F/dx² < 0: Unstable (local maximum)

    Args:
        F_func: Function F(x, T) -> scalar
        x_range: Range of x values
        T_range: Range of temperatures
        n_x: Grid resolution in x
        n_T: Grid resolution in T

    Returns:
        Dictionary with:
        - 'x': x grid values
        - 'T': T grid values
        - 'stability_map': 2D array with +1 (stable), -1 (unstable)
        - 'd2F': Second derivative values
    """
    x_grid, T_grid, d2F = find_bifurcation_curve(F_func, x_range, T_range, n_x, n_T)

    stability_map = jnp.sign(d2F)

    return {
        "x": x_grid,
        "T": T_grid,
        "stability_map": stability_map,
        "d2F": d2F,
    }


def find_spinodal_points(
    d2F: jnp.ndarray,
    x_grid: jnp.ndarray,
    T_fixed: float,
    T_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Find spinodal points (d²F/dx² = 0) at a fixed temperature.

    Args:
        d2F: Second derivative grid, shape (n_x, n_T)
        x_grid: x values
        T_fixed: Temperature of interest
        T_grid: Temperature grid

    Returns:
        x values where d²F/dx² = 0
    """
    # Find nearest T index
    T_idx = jnp.argmin(jnp.abs(T_grid - T_fixed))

    # Get d2F slice at this T
    d2F_slice = d2F[:, T_idx]

    # Find sign changes (zero crossings)
    from pycse.sklearn.zenn.analysis.derivatives import find_zero_crossings

    return find_zero_crossings(d2F_slice, x_grid)


def create_2D_F_function(
    model,
    params: Dict[str, Any],
    kb: float = 1.0,
    gamma: float = 100.0,
) -> Callable:
    """
    Create F(x, T) function from a trained ZENN model.

    Args:
        model: ZENN model
        params: Model parameters
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale

    Returns:
        Function F(x, T) -> scalar
    """
    from pycse.sklearn.zenn.utils.thermodynamics import compute_total_helmholtz_energy

    def F_func(x, T):
        x = jnp.atleast_1d(x).reshape(1, -1)
        T_arr = jnp.array([T])
        outputs = model.apply(params, x, T_arr)
        return compute_total_helmholtz_energy(
            outputs["E"], outputs["S"], T_arr, kb, gamma
        )[0]

    return F_func
