"""
Learnable temperature module for ZENN (Algorithm 1).

Implements the EM algorithm for discovering latent temperature modes
in heterogeneous datasets.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional, Dict, Any


class LearnableTemperatureSet(nn.Module):
    """
    Learnable temperature set for multi-source data integration.

    Maintains K learnable temperatures that capture latent heterogeneity
    in the dataset. Uses sigmoid parameterization to ensure temperatures
    stay within bounds.

    Attributes:
        n_temperatures: Number of learnable temperatures
        T_min: Minimum temperature bound
        T_max: Maximum temperature bound
        include_fixed_T1: Whether to include a fixed T=1.0 reference
    """

    n_temperatures: int = 4
    T_min: float = 0.1
    T_max: float = 10.0
    include_fixed_T1: bool = True

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """
        Return the temperature set.

        Returns:
            Temperature values, shape (n_temperatures,) or (n_temperatures + 1,)
        """
        # Learnable raw parameters (before sigmoid transformation)
        raw_temps = self.param(
            "raw_temperatures",
            nn.initializers.normal(stddev=0.1),
            (self.n_temperatures,),
        )

        # Transform to bounded range: T = T_min + (T_max - T_min) * sigmoid(raw)
        temperatures = self.T_min + (self.T_max - self.T_min) * jax.nn.sigmoid(
            raw_temps
        )

        if self.include_fixed_T1:
            # Include fixed T=1.0 reference
            temperatures = jnp.concatenate([jnp.array([1.0]), temperatures])

        return temperatures


def compute_temperature_posterior(
    cz_losses: jnp.ndarray,
    omega: float = 5.0,
) -> jnp.ndarray:
    """
    Compute posterior probability over temperatures (Eq. 11).

    q(T_i | x, y) = exp(-omega * CZ(x, y, T_i)) / sum_j exp(-omega * CZ(x, y, T_j))

    Args:
        cz_losses: Cross-zentropy losses at each temperature,
                   shape (n_temps,) for single sample or (n_temps, batch_size) for batch
        omega: Temperature selection sharpness
               - Use omega=5.0 during training (emphasizes worst-case)
               - Use omega=1.0 during testing (true posterior)

    Returns:
        Posterior probabilities, same shape as cz_losses
    """
    log_q = -omega * cz_losses
    if cz_losses.ndim == 1:
        log_q = log_q - jax.scipy.special.logsumexp(log_q)
    else:
        log_q = log_q - jax.scipy.special.logsumexp(log_q, axis=0, keepdims=True)
    return jnp.exp(log_q)


def em_temperature_step(
    model_params: Dict[str, Any],
    model_apply_fn,
    X: jnp.ndarray,
    y: jnp.ndarray,
    temperatures: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    omega: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single EM step for temperature learning (Algorithm 1).

    E-step: Compute posterior q(T|x,y) for each sample
    Returns weighted loss for M-step optimization

    Args:
        model_params: ZENN model parameters
        model_apply_fn: Function to apply model: apply_fn(params, x, T) -> outputs
        X: Input features, shape (batch_size, n_features)
        y: Labels (one-hot), shape (batch_size, n_classes)
        temperatures: Temperature set, shape (n_temps,)
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        omega: Posterior sharpness

    Returns:
        Tuple of (weighted_loss, temperature_posterior)
    """
    from pycse.sklearn.zenn.losses.cross_zentropy import cross_zentropy_loss

    batch_size = X.shape[0]
    n_temps = temperatures.shape[0]

    # Compute CZ loss at each temperature
    def cz_at_temp(T):
        outputs = model_apply_fn(model_params, X, jnp.full((batch_size,), T))
        E, S = outputs["E"], outputs["S"]
        return cross_zentropy_loss(E, S, y, T, kb, gamma, reduction="none")

    # Shape: (n_temps, batch_size)
    cz_losses = jax.vmap(cz_at_temp)(temperatures)

    # E-step: compute posterior q(T|x,y)
    q = compute_temperature_posterior(cz_losses, omega)

    # Weighted loss for M-step
    weighted_loss = jnp.sum(q * cz_losses, axis=0)  # (batch_size,)
    total_loss = jnp.mean(weighted_loss)

    return total_loss, q


def optimize_temperatures(
    model_params: Dict[str, Any],
    model_apply_fn,
    X: jnp.ndarray,
    y: jnp.ndarray,
    initial_temperatures: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    omega: float = 5.0,
    n_steps: int = 10,
    lr: float = 1e-3,
) -> jnp.ndarray:
    """
    Optimize temperature values given fixed model parameters (M-step for T).

    Args:
        model_params: Fixed ZENN model parameters
        model_apply_fn: Function to apply model
        X: Input features
        y: Labels
        initial_temperatures: Starting temperature values
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        omega: Posterior sharpness
        n_steps: Number of optimization steps
        lr: Learning rate

    Returns:
        Optimized temperatures
    """
    import optax

    def temperature_loss(raw_temps):
        # Transform raw parameters to bounded temperatures
        temps = 0.1 + 9.9 * jax.nn.sigmoid(raw_temps)
        loss, _ = em_temperature_step(
            model_params, model_apply_fn, X, y, temps, kb, gamma, omega
        )
        return loss

    # Initialize from current temperatures (inverse sigmoid transform)
    raw_temps = jnp.log((initial_temperatures - 0.1) / (10.0 - initial_temperatures))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(raw_temps)

    for _ in range(n_steps):
        loss, grads = jax.value_and_grad(temperature_loss)(raw_temps)
        updates, opt_state = optimizer.update(grads, opt_state)
        raw_temps = optax.apply_updates(raw_temps, updates)

    # Transform back to bounded temperatures
    optimized_temps = 0.1 + 9.9 * jax.nn.sigmoid(raw_temps)
    return optimized_temps


def get_map_temperature(
    q: jnp.ndarray,
    temperatures: jnp.ndarray,
) -> jnp.ndarray:
    """
    Get maximum a posteriori (MAP) temperature assignment for each sample.

    Args:
        q: Temperature posterior, shape (n_temps, batch_size)
        temperatures: Temperature values, shape (n_temps,)

    Returns:
        MAP temperature for each sample, shape (batch_size,)
    """
    map_indices = jnp.argmax(q, axis=0)
    return temperatures[map_indices]


def get_temperature_distribution(
    q: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Analyze the temperature distribution across samples.

    Args:
        q: Temperature posterior, shape (n_temps, batch_size)

    Returns:
        Dictionary with statistics about temperature assignments
    """
    # Mode assignment (MAP)
    mode_counts = jnp.sum(jnp.eye(q.shape[0])[jnp.argmax(q, axis=0)], axis=0)

    # Expected temperature per sample
    # Would need temperatures array for this

    return {
        "mode_counts": mode_counts,
        "mode_fractions": mode_counts / q.shape[1],
        "entropy": -jnp.sum(q * jnp.log(q + 1e-10), axis=0),  # Per-sample entropy
    }


def marginalize_over_temperatures(
    probs_at_temps: jnp.ndarray,
    q: jnp.ndarray,
) -> jnp.ndarray:
    """
    Marginalize predictions over temperature posterior.

    p(y|x) = sum_T q(T|x) * p(y|x,T)

    Args:
        probs_at_temps: Class probabilities at each temperature,
                        shape (n_temps, batch_size, n_classes)
        q: Temperature posterior, shape (n_temps, batch_size)

    Returns:
        Marginalized probabilities, shape (batch_size, n_classes)
    """
    # q: (n_temps, batch_size) -> (n_temps, batch_size, 1)
    # probs_at_temps: (n_temps, batch_size, n_classes)
    return jnp.sum(q[:, :, None] * probs_at_temps, axis=0)
