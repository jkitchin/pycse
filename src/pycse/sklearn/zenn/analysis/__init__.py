"""Analysis tools for ZENN models."""

from pycse.sklearn.zenn.analysis.derivatives import (
    compute_gradient,
    compute_hessian,
    compute_derivatives,
)
from pycse.sklearn.zenn.analysis.critical_points import (
    find_critical_point,
    find_bifurcation_curve,
)
from pycse.sklearn.zenn.analysis.uncertainty import (
    configuration_entropy,
    predictive_entropy,
    temperature_posterior_entropy,
    free_energy_margin,
    epistemic_uncertainty,
    aleatoric_uncertainty,
    total_uncertainty,
    mixing_entropy,
    confidence_score,
    num_effective_configs,
    uncertainty_calibration,
    ood_scores,
    selective_prediction,
)

__all__ = [
    "compute_gradient",
    "compute_hessian",
    "compute_derivatives",
    "find_critical_point",
    "find_bifurcation_curve",
    # Uncertainty quantification
    "configuration_entropy",
    "predictive_entropy",
    "temperature_posterior_entropy",
    "free_energy_margin",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
    "total_uncertainty",
    "mixing_entropy",
    "confidence_score",
    "num_effective_configs",
    "uncertainty_calibration",
    "ood_scores",
    "selective_prediction",
]
