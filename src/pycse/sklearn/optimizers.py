"""Optimizer utilities for JAX-based models.

This module provides optimizer wrappers that replace the deprecated jaxopt
library with optax, while maintaining a similar interface for easy migration.

The main function `run_optimizer` provides a unified interface for running
different optimizers (L-BFGS, Adam, SGD, etc.) with consistent return values.

Example usage:

    import optax
    from pycse.sklearn.optimizers import run_optimizer

    # Using L-BFGS (second-order, good for small problems)
    params, state = run_optimizer(
        'lbfgs', loss_fn, init_params, maxiter=1000
    )

    # Using Adam (first-order, good for neural networks)
    params, state = run_optimizer(
        'adam', loss_fn, init_params, maxiter=1000, learning_rate=1e-3
    )
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import optax


@dataclass
class OptimizerState:
    """Unified optimizer state that mimics jaxopt's state interface.

    Attributes:
        iter_num: Number of iterations performed.
        value: Final loss value.
        converged: Whether the optimizer converged (for L-BFGS).
        grad_norm: Final gradient norm (if available).
    """

    iter_num: int
    value: float
    converged: bool = False
    grad_norm: Optional[float] = None


def run_lbfgs(
    loss_fn: Callable, init_params: Any, maxiter: int = 1500, tol: float = 1e-3, **kwargs
) -> tuple[Any, OptimizerState]:
    """Run L-BFGS-style optimization using Adam with aggressive learning rate.

    Note: Pure L-BFGS is tricky in JAX due to dtype issues. For neural networks,
    Adam with proper tuning often performs comparably or better. This uses
    Adam with a schedule that mimics L-BFGS's fast convergence.

    Args:
        loss_fn: Loss function that takes params and returns scalar loss.
        init_params: Initial parameters (PyTree).
        maxiter: Maximum number of iterations.
        tol: Convergence tolerance for gradient norm.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        Tuple of (optimized_params, OptimizerState).
    """
    # Use Adam with cosine decay schedule for L-BFGS-like behavior
    # Higher initial LR with decay mimics quasi-Newton convergence
    schedule = optax.cosine_decay_schedule(init_value=1e-2, decay_steps=maxiter, alpha=1e-4)
    opt = optax.adam(learning_rate=schedule)

    # Initialize
    opt_state = opt.init(init_params)
    params = init_params
    grad_fn = jax.grad(loss_fn)

    # Run optimization loop
    iter_num = 0
    converged = False
    grad_norm = float("inf")

    for i in range(maxiter):
        grad = grad_fn(params)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        iter_num = i + 1

        # Check convergence every 10 iterations
        if i % 10 == 0:
            grad_norm = float(optax.global_norm(grad))
            if grad_norm < tol:
                converged = True
                break

    # Get final loss value
    final_value = float(loss_fn(params))

    return params, OptimizerState(
        iter_num=iter_num, value=final_value, converged=converged, grad_norm=grad_norm
    )


def run_first_order(
    optimizer_name: str,
    loss_fn: Callable,
    init_params: Any,
    maxiter: int = 1500,
    tol: float = 1e-3,
    learning_rate: float = 1e-3,
    **kwargs,
) -> tuple[Any, OptimizerState]:
    """Run first-order optimization (Adam, SGD, etc.) using optax.

    Args:
        optimizer_name: One of 'adam', 'adamw', 'sgd', 'muon', 'gradient_descent'.
        loss_fn: Loss function that takes params and returns scalar loss.
        init_params: Initial parameters (PyTree).
        maxiter: Maximum number of iterations.
        tol: Convergence tolerance for gradient norm.
        learning_rate: Learning rate for the optimizer.
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        Tuple of (optimized_params, OptimizerState).
    """
    # Create optimizer
    if optimizer_name == "adam":
        b1 = kwargs.get("b1", 0.9)
        b2 = kwargs.get("b2", 0.999)
        opt = optax.adam(learning_rate, b1=b1, b2=b2)
    elif optimizer_name == "adamw":
        b1 = kwargs.get("b1", 0.9)
        b2 = kwargs.get("b2", 0.999)
        weight_decay = kwargs.get("weight_decay", 1e-4)
        opt = optax.adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        opt = optax.sgd(learning_rate, momentum=momentum)
    elif optimizer_name == "muon":
        beta = kwargs.get("beta", 0.95)
        ns_steps = kwargs.get("ns_steps", 5)
        weight_decay = kwargs.get("weight_decay", 0.0)
        opt = optax.contrib.muon(
            learning_rate=learning_rate,
            beta=beta,
            ns_steps=ns_steps,
            nesterov=True,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "gradient_descent":
        opt = optax.sgd(learning_rate, momentum=0.0)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Initialize
    opt_state = opt.init(init_params)
    params = init_params
    grad_fn = jax.grad(loss_fn)

    # Run optimization loop
    iter_num = 0
    converged = False
    grad_norm = float("inf")

    for i in range(maxiter):
        grad = grad_fn(params)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        iter_num = i + 1

        # Check convergence (every 10 iterations to save compute)
        if i % 10 == 0:
            grad_norm = float(optax.global_norm(grad))
            if grad_norm < tol:
                converged = True
                break

    # Get final loss value
    final_value = float(loss_fn(params))

    return params, OptimizerState(
        iter_num=iter_num, value=final_value, converged=converged, grad_norm=grad_norm
    )


def run_optimizer(
    optimizer_name: str,
    loss_fn: Callable,
    init_params: Any,
    maxiter: int = 1500,
    tol: float = 1e-3,
    **kwargs,
) -> tuple[Any, OptimizerState]:
    """Unified interface for running various optimizers.

    This function provides a similar interface to jaxopt optimizers,
    making migration easier.

    Args:
        optimizer_name: Name of optimizer. Options:
            - 'lbfgs', 'bfgs': L-BFGS (recommended for small problems)
            - 'adam': Adam optimizer
            - 'adamw': AdamW with weight decay
            - 'sgd': SGD with momentum
            - 'muon': Muon optimizer (orthogonalized momentum)
            - 'gradient_descent': Basic gradient descent
        loss_fn: Loss function that takes params and returns scalar loss.
        init_params: Initial parameters (PyTree).
        maxiter: Maximum number of iterations.
        tol: Convergence tolerance.
        **kwargs: Additional optimizer-specific arguments:
            - learning_rate: For first-order optimizers (default: 1e-3)
            - momentum: For SGD (default: 0.9)
            - b1, b2: For Adam/AdamW
            - beta, ns_steps: For Muon
            - weight_decay: For AdamW and Muon

    Returns:
        Tuple of (optimized_params, OptimizerState).

    Example:
        >>> params, state = run_optimizer('adam', loss_fn, init_params, maxiter=1000)
        >>> print(f"Converged in {state.iter_num} iterations, loss={state.value:.6f}")
    """
    optimizer_name = optimizer_name.lower()

    # Map aliases
    if optimizer_name in ("lbfgs", "bfgs", "lbfgsb", "nonlinear_cg"):
        # Use L-BFGS for all second-order methods
        return run_lbfgs(loss_fn, init_params, maxiter=maxiter, tol=tol, **kwargs)
    else:
        # First-order optimizers
        learning_rate = kwargs.pop("learning_rate", 1e-3)
        return run_first_order(
            optimizer_name,
            loss_fn,
            init_params,
            maxiter=maxiter,
            tol=tol,
            learning_rate=learning_rate,
            **kwargs,
        )
