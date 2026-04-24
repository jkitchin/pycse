"""
Cross-zentropy loss function for classification tasks (Eq. 9).

This extends cross-entropy loss by incorporating intrinsic entropy contributions
from zentropy theory, enabling better handling of heterogeneous data.
"""

import jax
import jax.numpy as jnp
from typing import Optional

from pycse.sklearn.zenn.utils.thermodynamics import compute_configuration_probabilities


def cross_zentropy_loss(
    E: jnp.ndarray,
    S: jnp.ndarray,
    y: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> jnp.ndarray:
    """
    Compute cross-zentropy loss for classification (Eq. 9).

    L_CZ = (1/M) * sum_j [y_j · (E - T*S)/(kb*T) + (S/(gamma*kb))^2 + ln(Z)]

    When S = 0, this reduces to standard cross-entropy loss.

    Args:
        E: Internal energy predictions, shape (batch_size, n_classes)
        S: Entropy predictions, shape (batch_size, n_classes)
        y: One-hot encoded labels, shape (batch_size, n_classes)
        T: Temperature, scalar or shape (batch_size,)
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        label_smoothing: Label smoothing factor (0-1)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value (scalar if reduction='mean' or 'sum', else (batch_size,))
    """
    n_classes = y.shape[-1]

    # Apply label smoothing if specified
    if label_smoothing > 0:
        y = y * (1 - label_smoothing) + label_smoothing / n_classes

    # Compute configuration probabilities
    p, log_p = compute_configuration_probabilities(E, S, T, kb, gamma)

    # Cross-zentropy: negative log-likelihood of correct class
    # L = -sum_k y_k * log(p_k)
    loss = -jnp.sum(y * log_p, axis=-1)

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        return loss


def cross_zentropy_loss_with_temperature_posterior(
    E: jnp.ndarray,
    S: jnp.ndarray,
    y: jnp.ndarray,
    temperatures: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    omega: float = 5.0,
    reduction: str = "mean",
) -> jnp.ndarray:
    """
    Cross-zentropy loss with EM-style temperature posterior weighting (Eq. 10).

    L = (1/M) * sum_j sum_i q_j(T_i) * CZ(j, i)

    where q_j(T_i) is the posterior probability of temperature T_i for sample j.

    Args:
        E: Energy predictions, shape (batch_size, n_classes)
        S: Entropy predictions, shape (batch_size, n_classes)
        y: One-hot labels, shape (batch_size, n_classes)
        temperatures: Temperature set, shape (n_temps,)
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        omega: Temperature selection sharpness (5.0 for training, 1.0 for test)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    batch_size = E.shape[0]
    n_temps = temperatures.shape[0]

    # Compute CZ loss for each temperature
    def cz_at_temp(T):
        return cross_zentropy_loss(E, S, y, T, kb, gamma, reduction="none")

    # Shape: (n_temps, batch_size)
    cz_losses = jax.vmap(cz_at_temp)(temperatures)

    # Compute posterior q(T|x,y) via Eq. 11
    # q_j(T_i) = exp(-omega * CZ(j,i)) / sum_i' exp(-omega * CZ(j,i'))
    log_q = -omega * cz_losses
    log_q = log_q - jax.scipy.special.logsumexp(log_q, axis=0, keepdims=True)
    q = jnp.exp(log_q)  # Shape: (n_temps, batch_size)

    # Weighted loss: sum over temperatures
    weighted_loss = jnp.sum(q * cz_losses, axis=0)  # Shape: (batch_size,)

    if reduction == "mean":
        return jnp.mean(weighted_loss)
    elif reduction == "sum":
        return jnp.sum(weighted_loss)
    else:
        return weighted_loss


def cross_entropy_loss(
    logits: jnp.ndarray,
    y: jnp.ndarray,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> jnp.ndarray:
    """
    Standard cross-entropy loss (baseline for comparison).

    This is equivalent to cross_zentropy_loss with S = 0.

    Args:
        logits: Raw model outputs, shape (batch_size, n_classes)
        y: One-hot encoded labels
        label_smoothing: Label smoothing factor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    n_classes = y.shape[-1]

    if label_smoothing > 0:
        y = y * (1 - label_smoothing) + label_smoothing / n_classes

    log_p = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(y * log_p, axis=-1)

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        return loss


def compute_accuracy(
    E: jnp.ndarray,
    S: jnp.ndarray,
    y: jnp.ndarray,
    T: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> jnp.ndarray:
    """
    Compute classification accuracy.

    Args:
        E: Energy predictions
        S: Entropy predictions
        y: One-hot labels
        T: Temperature
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale

    Returns:
        Accuracy (scalar between 0 and 1)
    """
    p, _ = compute_configuration_probabilities(E, S, T, kb, gamma)
    predictions = jnp.argmax(p, axis=-1)
    targets = jnp.argmax(y, axis=-1)
    return jnp.mean(predictions == targets)


def compute_accuracy_with_temperature_marginalization(
    E: jnp.ndarray,
    S: jnp.ndarray,
    y: jnp.ndarray,
    temperatures: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
    omega: float = 1.0,
) -> jnp.ndarray:
    """
    Compute accuracy with temperature marginalization for heterogeneous data.

    Predictions are computed by marginalizing over the temperature posterior:
    p(y|x) = sum_T q(T|x) * p(y|x,T)

    Args:
        E: Energy predictions
        S: Entropy predictions
        y: One-hot labels
        temperatures: Temperature set
        kb: Boltzmann constant
        gamma: Entropy fluctuation scale
        omega: Posterior sharpness (1.0 for testing)

    Returns:
        Accuracy
    """

    def probs_at_temp(T):
        p, log_p = compute_configuration_probabilities(E, S, T, kb, gamma)
        return p, log_p

    # Get probabilities at each temperature
    # Shape: (n_temps, batch_size, n_classes)
    all_probs, all_log_probs = jax.vmap(probs_at_temp)(temperatures)

    # Compute cross-zentropy loss at each temperature for posterior
    def cz_at_temp(T):
        return cross_zentropy_loss(E, S, y, T, kb, gamma, reduction="none")

    cz_losses = jax.vmap(cz_at_temp)(temperatures)  # (n_temps, batch_size)

    # Temperature posterior q(T|x)
    log_q = -omega * cz_losses
    log_q = log_q - jax.scipy.special.logsumexp(log_q, axis=0, keepdims=True)
    q = jnp.exp(log_q)  # (n_temps, batch_size)

    # Marginalize: p(y|x) = sum_T q(T|x) * p(y|x,T)
    # q: (n_temps, batch_size) -> (n_temps, batch_size, 1)
    # all_probs: (n_temps, batch_size, n_classes)
    marginalized_probs = jnp.sum(
        q[:, :, None] * all_probs, axis=0
    )  # (batch_size, n_classes)

    predictions = jnp.argmax(marginalized_probs, axis=-1)
    targets = jnp.argmax(y, axis=-1)
    return jnp.mean(predictions == targets)
