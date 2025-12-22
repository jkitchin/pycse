"""Sklearn-compatible estimators for pycse.

This module provides sklearn-compatible estimators with uncertainty
quantification capabilities.

Available estimators:
- DPOSE: Direct Propagation of Shallow Ensembles (JAX/Flax)
- KAN: Kolmogorov-Arnold Networks (JAX/Flax)
- KfoldNN: K-fold ensemble neural network (JAX/Flax)
- LLPR: Last-Layer Prediction Rigidity (JAX/Flax)
- NNBR: Neural Network + Bayesian Ridge (sklearn)
- NNGMM: Neural Network + Gaussian Mixture Model (sklearn)
- LeafModelRegressor: Decision tree with sub-models per leaf
"""

# Lazy imports to avoid loading all backends
__all__ = [
    "DPOSE",
    "KAN",
    "KfoldNN",
    "LLPR",
    "NNBR",
    "NNGMM",
    "LeafModelRegressor",
]


def __getattr__(name):
    """Lazy import of estimators."""
    if name == "DPOSE":
        from pycse.sklearn.dpose import DPOSE

        return DPOSE
    elif name == "KAN":
        from pycse.sklearn.kan import KAN

        return KAN
    elif name == "KfoldNN":
        from pycse.sklearn.kfoldnn import KfoldNN

        return KfoldNN
    elif name == "LLPR":
        from pycse.sklearn.llpr_regressor import LLPR

        return LLPR
    elif name == "NNBR":
        from pycse.sklearn.nnbr import NNBR

        return NNBR
    elif name == "NNGMM":
        from pycse.sklearn.nngmm import NNGMM

        return NNGMM
    elif name == "LeafModelRegressor":
        from pycse.sklearn.leaf_model import LeafModelRegressor

        return LeafModelRegressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
