"""Sklearn-compatible estimators for pycse.

This module provides sklearn-compatible estimators with uncertainty
quantification capabilities.

Available estimators:
- DPOSE: Direct Propagation of Shallow Ensembles (JAX/Flax)
- JAXICNNRegressor: Input Convex Neural Network for convex surrogates (JAX)
- JAXMonotonicRegressor: Monotonic Neural Network with LLPR uncertainty (JAX)
- JAXPeriodicRegressor: Periodic Neural Network with LLPR uncertainty (JAX)
- KAN: Kolmogorov-Arnold Networks (JAX/Flax)
- KANLLPR: KAN with Last-Layer Prediction Rigidity (JAX/Flax)
- KfoldNN: K-fold ensemble neural network (JAX/Flax)
- LLPR / LLPRRegressor: Last-Layer Prediction Rigidity (JAX/Flax)
- NNBR / NeuralNetworkBLR: Neural Network + Bayesian Ridge (sklearn)
- NNGMM / NeuralNetworkGMM: Neural Network + Gaussian Mixture Model (sklearn)
- LinearRegressionUQ: Linear regression with parameter/prediction uncertainty
- NFlowsRegressor: Normalizing-flow regressor (optional: torch, nflows)
- LeafModelRegressor: Decision tree with sub-models per leaf
- SISSO: Sure Independence Screening and Sparsifying Operator (TorchSISSO)
- SISSOEnsemble: Shallow ensemble of SISSO equations with calibrated UQ
- ActiveLearner: Model-agnostic active learning for iterative experiment selection
- ZENNClassifier: Zentropy-Enhanced Neural Network classifier (JAX/Flax)
- ZENNRegressor: Zentropy-Enhanced Neural Network regressor (JAX/Flax)
- ZENNRegressorNLL: ZENN regressor with negative-log-likelihood head (JAX/Flax)

Short aliases (``LLPR``, ``NNBR``, ``NNGMM``) refer to the same classes as
their full names.
"""

import importlib

# Lazy imports to avoid loading all backends (JAX, torch, ...) on import.
# Maps public name -> (module, attribute).
_LAZY = {
    "ActiveLearner": ("pycse.sklearn.active_learning", "ActiveLearner"),
    "DPOSE": ("pycse.sklearn.dpose", "DPOSE"),
    "JAXICNNRegressor": ("pycse.sklearn.jax_icnn", "JAXICNNRegressor"),
    "JAXMonotonicRegressor": ("pycse.sklearn.jax_monotonic", "JAXMonotonicRegressor"),
    "JAXPeriodicRegressor": ("pycse.sklearn.jax_periodic", "JAXPeriodicRegressor"),
    "KAN": ("pycse.sklearn.kan", "KAN"),
    "KANLLPR": ("pycse.sklearn.kan_llpr", "KANLLPR"),
    "KfoldNN": ("pycse.sklearn.kfoldnn", "KfoldNN"),
    "LLPR": ("pycse.sklearn.llpr_regressor", "LLPRRegressor"),
    "LLPRRegressor": ("pycse.sklearn.llpr_regressor", "LLPRRegressor"),
    "NNBR": ("pycse.sklearn.nnbr", "NeuralNetworkBLR"),
    "NeuralNetworkBLR": ("pycse.sklearn.nnbr", "NeuralNetworkBLR"),
    "NNGMM": ("pycse.sklearn.nngmm", "NeuralNetworkGMM"),
    "NeuralNetworkGMM": ("pycse.sklearn.nngmm", "NeuralNetworkGMM"),
    "LeafModelRegressor": ("pycse.sklearn.leaf_model", "LeafModelRegressor"),
    "LinearRegressionUQ": ("pycse.sklearn.lr_uq", "LinearRegressionUQ"),
    "NFlowsRegressor": ("pycse.sklearn.nflows_regressor", "NFlowsRegressor"),
    "SISSO": ("pycse.sklearn.sisso", "SISSO"),
    "SISSOEnsemble": ("pycse.sklearn.sisso", "SISSOEnsemble"),
    "ZENNClassifier": ("pycse.sklearn.zenn", "ZENNClassifier"),
    "ZENNRegressor": ("pycse.sklearn.zenn", "ZENNRegressor"),
    "ZENNRegressorNLL": ("pycse.sklearn.zenn.estimators.regressor_nll", "ZENNRegressorNLL"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    """Lazy import of estimators."""
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    obj = getattr(importlib.import_module(module_name), attr)
    globals()[name] = obj  # cache so later lookups skip __getattr__
    return obj


def __dir__():
    """Include lazily exported names in dir()."""
    return sorted(set(globals()) | set(__all__))
