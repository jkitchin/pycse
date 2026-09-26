"""Optimizer utilities for JAX-based models.

This module provides optimizer wrappers that replace the deprecated jaxopt
library with optax, while maintaining a similar interface for easy migration.

The main function `run_optimizer` provides a unified interface for running
different optimizers (Adam, SGD, Muon, ...) with consistent return values.

``'lbfgs'`` and ``'bfgs'`` run a true limited-memory BFGS (``optax.lbfgs`` with a
zoom line search satisfying the strong Wolfe conditions). There is no full-matrix
BFGS; ``'bfgs'`` is an alias for L-BFGS. The legacy jaxopt names ``'lbfgsb'`` and
``'nonlinear_cg'`` also run L-BFGS and emit a ``UserWarning`` (no box constraints,
no conjugate gradient). ``'adam_cosine'`` is the Adam + cosine-decay schedule that
``'lbfgs'``/``'bfgs'`` used to run, for reproducing older results.

Example usage:

    import optax
    from pycse.sklearn.optimizers import run_optimizer

    # L-BFGS (quasi-Newton with line search; good for small/medium smooth problems)
    params, state = run_optimizer(
        'lbfgs', loss_fn, init_params, maxiter=1000
    )

    # Using Adam (first-order, good for neural networks)
    params, state = run_optimizer(
        'adam', loss_fn, init_params, maxiter=1000, learning_rate=1e-3
    )
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu


@dataclass
class OptimizerState:
    """Unified optimizer state that mimics jaxopt's state interface.

    Attributes:
        iter_num: Number of iterations performed.
        value: Final loss value.
        converged: Whether the gradient-norm tolerance was reached.
        grad_norm: Final gradient norm (if available).
    """

    iter_num: int
    value: float
    converged: bool = False
    grad_norm: Optional[float] = None


def run_lbfgs(
    loss_fn: Callable,
    init_params: Any,
    maxiter: int = 1500,
    tol: float = 1e-3,
    memory_size: int = 10,
    **kwargs,
) -> tuple[Any, OptimizerState]:
    """Minimize ``loss_fn`` with L-BFGS (``optax.lbfgs``, zoom line search).

    The whole loop runs under ``jax.jit`` with ``jax.lax.while_loop``, so
    ``loss_fn`` must be traceable by JAX. It stops when the global gradient norm
    drops below ``tol``, after ``maxiter`` iterations, or when the loss becomes
    non-finite (then the last finite parameters are returned).

    Args:
        loss_fn: Loss function that takes params and returns scalar loss.
        init_params: Initial parameters (PyTree).
        maxiter: Maximum number of iterations.
        tol: Convergence tolerance for the global gradient norm.
        memory_size: Number of curvature pairs kept by L-BFGS.
        **kwargs: Additional arguments (ignored for compatibility, e.g. a
            ``learning_rate`` meant for first-order optimizers).

    Returns:
        Tuple of (optimized_params, OptimizerState).
    """
    opt = optax.lbfgs(memory_size=memory_size)
    # optax stores the loss in its state with the default float dtype (float64
    # when jax_enable_x64 is on); cast the loss to match, or lax.cond rejects it.
    value_dtype = otu.tree_get(opt.init(init_params), "value").dtype
    user_loss = loss_fn

    def loss_fn(params):
        return jnp.asarray(user_loss(params), dtype=value_dtype)

    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def step(carry):
        params, state, _ = carry
        value, grad = value_and_grad(params, state=state)
        updates, new_state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=loss_fn
        )
        return optax.apply_updates(params, updates), new_state, params

    def keep_going(carry):
        _, state, _ = carry
        count = otu.tree_get(state, "count")
        grad_norm = optax.global_norm(otu.tree_get(state, "grad"))
        value = otu.tree_get(state, "value")
        return (count == 0) | ((count < maxiter) & (grad_norm >= tol) & jnp.isfinite(value))

    @jax.jit
    def solve(params):
        carry = (params, opt.init(params), params)
        return jax.lax.while_loop(keep_going, step, carry)

    params, state, prev_params = solve(init_params)
    final_value = float(loss_fn(params))
    if not jnp.isfinite(final_value):
        params = prev_params
        final_value = float(loss_fn(params))
    grad_norm = float(optax.global_norm(jax.grad(loss_fn)(params)))

    return params, OptimizerState(
        iter_num=int(otu.tree_get(state, "count")),
        value=final_value,
        converged=grad_norm < tol,
        grad_norm=grad_norm,
    )


def run_adam_cosine(
    loss_fn: Callable, init_params: Any, maxiter: int = 1500, tol: float = 1e-3, **kwargs
) -> tuple[Any, OptimizerState]:
    """Adam with a cosine-decay learning rate (1e-2 decaying to 1e-6).

    This is the schedule that ``'lbfgs'``/``'bfgs'`` used to run; it is
    available as ``'adam_cosine'`` in :func:`run_optimizer`.

    Args:
        loss_fn: Loss function that takes params and returns scalar loss.
        init_params: Initial parameters (PyTree).
        maxiter: Maximum number of iterations.
        tol: Convergence tolerance for gradient norm.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        Tuple of (optimized_params, OptimizerState).
    """
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


# Legacy jaxopt optimizer names that are accepted but not implemented as such.
_UNSUPPORTED_ALIASES = {
    "lbfgsb": "box constraints are not supported",
    "nonlinear_cg": "no nonlinear conjugate gradient is performed",
}


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
            - 'lbfgs', 'bfgs': L-BFGS with a line search (see :func:`run_lbfgs`;
              'bfgs' is an alias, there is no full-matrix BFGS)
            - 'lbfgsb', 'nonlinear_cg': accepted for backward compatibility
              (jaxopt names) and run as L-BFGS; a UserWarning is emitted
              (no box constraints, no conjugate gradient)
            - 'adam_cosine': Adam with a cosine-decay schedule from 1e-2 (what
              'lbfgs'/'bfgs' used to run; see :func:`run_adam_cosine`)
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
            - memory_size: For L-BFGS (default: 10)
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
    if optimizer_name in _UNSUPPORTED_ALIASES:
        warnings.warn(
            f"optimizer={optimizer_name!r} is not implemented as a distinct algorithm; "
            f"it runs L-BFGS like 'lbfgs' ({_UNSUPPORTED_ALIASES[optimizer_name]}).",
            UserWarning,
            stacklevel=2,
        )
    if optimizer_name in ("lbfgs", "bfgs") or optimizer_name in _UNSUPPORTED_ALIASES:
        return run_lbfgs(loss_fn, init_params, maxiter=maxiter, tol=tol, **kwargs)
    elif optimizer_name == "adam_cosine":
        return run_adam_cosine(loss_fn, init_params, maxiter=maxiter, tol=tol, **kwargs)
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
