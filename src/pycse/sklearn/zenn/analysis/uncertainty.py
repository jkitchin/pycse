"""
Uncertainty Quantification for ZENN.

ZENN provides natural uncertainty quantification through its thermodynamic framework:

1. Configuration probability entropy (epistemic) - Which configs explain the data?
2. Entropy networks S(k) (aleatoric) - Learned task-specific noise
3. Temperature posterior entropy (meta) - Which data source generated this?
4. Free energy mixing entropy (combined) - Total system uncertainty

This module provides functions to extract and analyze these uncertainty measures.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union


def configuration_entropy(
    p_config: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    Compute epistemic uncertainty via configuration probability entropy.

    H_config(x) = -sum_k p(k|x) * log(p(k|x))

    High entropy means multiple configurations are plausible (high epistemic uncertainty).
    Low entropy means one configuration dominates (low epistemic uncertainty).

    Parameters
    ----------
    p_config : array of shape (n_samples, n_configs)
        Configuration probabilities p(k) for each sample.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    H_config : array of shape (n_samples,)
        Configuration entropy for each sample.
    """
    log_p = jnp.log(p_config + eps)
    H = -jnp.sum(p_config * log_p, axis=-1)
    return H


def predictive_entropy(
    class_probs: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    Compute aleatoric uncertainty via class probability entropy.

    H_pred(x) = -sum_c p(c|x) * log(p(c|x))

    For classification, this measures confidence in the predicted class.

    Parameters
    ----------
    class_probs : array of shape (n_samples, n_classes)
        Class probabilities for each sample.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    H_pred : array of shape (n_samples,)
        Predictive entropy for each sample.
    """
    log_p = jnp.log(class_probs + eps)
    H = -jnp.sum(class_probs * log_p, axis=-1)
    return H


def temperature_posterior_entropy(
    q_temp: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    Compute meta-uncertainty via temperature posterior entropy.

    H_temp(x,y) = -sum_T q(T|x,y) * log(q(T|x,y))

    High entropy means the sample could belong to multiple temperature modes
    (ambiguous data source). Low entropy means clear source assignment.

    Parameters
    ----------
    q_temp : array of shape (n_temperatures, n_samples)
        Temperature posterior q(T|x,y) for each sample.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    H_temp : array of shape (n_samples,)
        Temperature posterior entropy for each sample.
    """
    log_q = jnp.log(q_temp + eps)
    H = -jnp.sum(q_temp * log_q, axis=0)
    return H


def free_energy_margin(
    F: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute confidence based on free energy gap between configurations.

    margin(x) = F_max - F_min

    Larger margin = more confident prediction (clear winner).
    Smaller margin = less confident (configurations are similar).

    Parameters
    ----------
    F : array of shape (n_samples, n_configs)
        Free energy F(k) for each configuration.

    Returns
    -------
    margin : array of shape (n_samples,)
        Free energy margin for each sample.
    """
    F_min = jnp.min(F, axis=-1)
    F_max = jnp.max(F, axis=-1)
    return F_max - F_min


def epistemic_uncertainty(
    E: jnp.ndarray,
    p_config: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute epistemic uncertainty as configuration disagreement in energy.

    sigma_epistemic = sqrt(sum_k p(k) * (E(k) - E_mean)^2)

    Measures how much configurations disagree about the energy value.

    Parameters
    ----------
    E : array of shape (n_samples, n_configs)
        Energy E(k) for each configuration.
    p_config : array of shape (n_samples, n_configs)
        Configuration probabilities.

    Returns
    -------
    sigma_epistemic : array of shape (n_samples,)
        Epistemic uncertainty (energy std across configs).
    """
    E_mean = jnp.sum(E * p_config, axis=-1, keepdims=True)
    variance = jnp.sum(p_config * (E - E_mean) ** 2, axis=-1)
    return jnp.sqrt(variance + 1e-10)


def aleatoric_uncertainty(
    S: jnp.ndarray,
    p_config: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute aleatoric uncertainty from entropy networks.

    U_aleatoric = sum_k p(k) * S(k)

    Expected intrinsic entropy across configurations.
    This captures inherent noise/uncertainty in the data.

    Parameters
    ----------
    S : array of shape (n_samples, n_configs)
        Entropy S(k) for each configuration.
    p_config : array of shape (n_samples, n_configs)
        Configuration probabilities.

    Returns
    -------
    U_aleatoric : array of shape (n_samples,)
        Aleatoric uncertainty (expected entropy).
    """
    return jnp.sum(p_config * S, axis=-1)


def total_uncertainty(
    E: jnp.ndarray,
    S: jnp.ndarray,
    p_config: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Decompose total uncertainty into epistemic and aleatoric components.

    Parameters
    ----------
    E : array of shape (n_samples, n_configs)
        Energy E(k) for each configuration.
    S : array of shape (n_samples, n_configs)
        Entropy S(k) for each configuration.
    p_config : array of shape (n_samples, n_configs)
        Configuration probabilities.

    Returns
    -------
    dict with:
        - 'epistemic': Configuration disagreement in energy
        - 'aleatoric': Expected entropy from S networks
        - 'total': Combined uncertainty (sqrt of sum of squares)
    """
    epist = epistemic_uncertainty(E, p_config)
    aleat = aleatoric_uncertainty(S, p_config)
    total = jnp.sqrt(epist ** 2 + aleat ** 2)

    return {
        'epistemic': epist,
        'aleatoric': aleat,
        'total': total,
    }


def mixing_entropy(
    p_config: jnp.ndarray,
    T: float,
    kb: float = 1.0,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    Compute the thermodynamic mixing entropy contribution to free energy.

    S_mix = -k_B * T * sum_k p(k) * ln(p(k))

    This is the statistical entropy from configuration mixing.
    Higher values indicate more uncertainty in the prediction.

    Parameters
    ----------
    p_config : array of shape (n_samples, n_configs)
        Configuration probabilities.
    T : float
        Temperature.
    kb : float
        Boltzmann constant.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    S_mix : array of shape (n_samples,)
        Mixing entropy contribution.
    """
    log_p = jnp.log(p_config + eps)
    return -kb * T * jnp.sum(p_config * log_p, axis=-1)


def confidence_score(
    p_config: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute confidence as maximum configuration probability.

    conf(x) = max_k p(k|x)

    Parameters
    ----------
    p_config : array of shape (n_samples, n_configs)
        Configuration probabilities.

    Returns
    -------
    conf : array of shape (n_samples,)
        Confidence score (0 to 1).
    """
    return jnp.max(p_config, axis=-1)


def num_effective_configs(
    p_config: jnp.ndarray,
    threshold: float = 0.01,
) -> jnp.ndarray:
    """
    Count the number of "active" configurations with significant probability.

    Parameters
    ----------
    p_config : array of shape (n_samples, n_configs)
        Configuration probabilities.
    threshold : float
        Minimum probability to count as active.

    Returns
    -------
    n_active : array of shape (n_samples,)
        Number of configurations with p > threshold.
    """
    return jnp.sum(p_config > threshold, axis=-1)


def uncertainty_calibration(
    predictions: np.ndarray,
    confidences: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute calibration metrics: does confidence match accuracy?

    Parameters
    ----------
    predictions : array of shape (n_samples,)
        Predicted class labels.
    confidences : array of shape (n_samples,)
        Confidence scores (max probability).
    true_labels : array of shape (n_samples,)
        True class labels.
    n_bins : int
        Number of bins for calibration curve.

    Returns
    -------
    dict with:
        - 'ece': Expected calibration error
        - 'mce': Maximum calibration error
        - 'bin_confidences': Mean confidence per bin
        - 'bin_accuracies': Accuracy per bin
        - 'bin_counts': Number of samples per bin
    """
    correct = (predictions == true_labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_confidences[i] = np.mean(confidences[mask])
            bin_accuracies[i] = np.mean(correct[mask])

    # Expected Calibration Error (weighted by bin size)
    weights = bin_counts / np.sum(bin_counts)
    ece = np.sum(weights * np.abs(bin_confidences - bin_accuracies))

    # Maximum Calibration Error
    mce = np.max(np.abs(bin_confidences - bin_accuracies))

    return {
        'ece': ece,
        'mce': mce,
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
    }


def ood_scores(
    H_in: np.ndarray,
    H_out: np.ndarray,
) -> Dict[str, float]:
    """
    Compute out-of-distribution detection metrics using uncertainty.

    Parameters
    ----------
    H_in : array of shape (n_in,)
        Uncertainty scores for in-distribution samples.
    H_out : array of shape (n_out,)
        Uncertainty scores for out-of-distribution samples.

    Returns
    -------
    dict with:
        - 'auroc': Area under ROC curve (higher = better OOD detection)
        - 'fpr95': False positive rate at 95% true positive rate
        - 'in_mean': Mean uncertainty for in-distribution
        - 'out_mean': Mean uncertainty for out-of-distribution
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Labels: 0 = in-distribution, 1 = out-of-distribution
    labels = np.concatenate([np.zeros(len(H_in)), np.ones(len(H_out))])
    scores = np.concatenate([H_in, H_out])

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    return {
        'auroc': auroc,
        'fpr95': fpr95,
        'in_mean': np.mean(H_in),
        'out_mean': np.mean(H_out),
    }


def uncertainty_vs_temperature(
    model,
    params: Dict[str, Any],
    X: jnp.ndarray,
    temperatures: jnp.ndarray,
    kb: float = 1.0,
    gamma: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Analyze how uncertainty changes with temperature.

    Parameters
    ----------
    model : ZENN model
        The trained model.
    params : dict
        Model parameters.
    X : array of shape (n_samples, n_features)
        Input data.
    temperatures : array of shape (n_temps,)
        Temperature values to analyze.
    kb : float
        Boltzmann constant.
    gamma : float
        Entropy fluctuation scale.

    Returns
    -------
    dict with arrays indexed by temperature:
        - 'config_entropy': Configuration entropy at each T
        - 'confidence': Max probability at each T
        - 'n_active': Number of active configs at each T
    """
    from pycse.sklearn.zenn.utils.thermodynamics import compute_configuration_probabilities

    results = {
        'temperatures': np.array(temperatures),
        'config_entropy': [],
        'confidence': [],
        'n_active': [],
    }

    for T in temperatures:
        T_arr = jnp.full((X.shape[0],), T)
        outputs = model.apply(params, X, T_arr)
        p, _ = compute_configuration_probabilities(
            outputs['E'], outputs['S'], T, kb, gamma
        )

        H = configuration_entropy(p)
        conf = confidence_score(p)
        n_active = num_effective_configs(p)

        results['config_entropy'].append(np.array(H))
        results['confidence'].append(np.array(conf))
        results['n_active'].append(np.array(n_active))

    # Stack into arrays
    results['config_entropy'] = np.stack(results['config_entropy'], axis=0)
    results['confidence'] = np.stack(results['confidence'], axis=0)
    results['n_active'] = np.stack(results['n_active'], axis=0)

    return results


def selective_prediction(
    predictions: np.ndarray,
    confidences: np.ndarray,
    true_labels: np.ndarray,
    coverage_levels: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate selective prediction: accuracy when rejecting uncertain samples.

    Parameters
    ----------
    predictions : array of shape (n_samples,)
        Predicted class labels.
    confidences : array of shape (n_samples,)
        Confidence scores.
    true_labels : array of shape (n_samples,)
        True class labels.
    coverage_levels : array, optional
        Coverage levels to evaluate. Default: [0.1, 0.2, ..., 1.0]

    Returns
    -------
    dict with:
        - 'coverage': Coverage levels evaluated
        - 'accuracy': Accuracy at each coverage level
        - 'risk': Error rate at each coverage level
    """
    if coverage_levels is None:
        coverage_levels = np.linspace(0.1, 1.0, 10)

    correct = (predictions == true_labels).astype(float)
    n_samples = len(predictions)

    # Sort by confidence (descending)
    order = np.argsort(-confidences)
    correct_sorted = correct[order]

    accuracies = []
    risks = []

    for coverage in coverage_levels:
        n_keep = int(coverage * n_samples)
        if n_keep > 0:
            acc = np.mean(correct_sorted[:n_keep])
            risk = 1.0 - acc
        else:
            acc = 1.0
            risk = 0.0
        accuracies.append(acc)
        risks.append(risk)

    return {
        'coverage': coverage_levels,
        'accuracy': np.array(accuracies),
        'risk': np.array(risks),
    }
