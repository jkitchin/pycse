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
- LLPR: Last-Layer Prediction Rigidity (JAX/Flax)
- NNBR: Neural Network + Bayesian Ridge (sklearn)
- NNGMM: Neural Network + Gaussian Mixture Model (sklearn)
- LeafModelRegressor: Decision tree with sub-models per leaf
- SISSO: Sure Independence Screening and Sparsifying Operator (TorchSISSO)
- SISSOEnsemble: Shallow ensemble of SISSO equations with calibrated UQ
"""

# Lazy imports to avoid loading all backends
__all__ = [
    "DPOSE",
    "JAXICNNRegressor",
    "JAXMonotonicRegressor",
    "JAXPeriodicRegressor",
    "KAN",
    "KANLLPR",
    "KfoldNN",
    "LLPR",
    "NNBR",
    "NNGMM",
    "LeafModelRegressor",
    "SISSO",
    "SISSOEnsemble",
]


def __getattr__(name):
    """Lazy import of estimators."""
    if name == "DPOSE":
        from pycse.sklearn.dpose import DPOSE

        return DPOSE
    elif name == "JAXICNNRegressor":
        from pycse.sklearn.jax_icnn import JAXICNNRegressor

        return JAXICNNRegressor
    elif name == "JAXMonotonicRegressor":
        from pycse.sklearn.jax_monotonic import JAXMonotonicRegressor

        return JAXMonotonicRegressor
    elif name == "JAXPeriodicRegressor":
        from pycse.sklearn.jax_periodic import JAXPeriodicRegressor

        return JAXPeriodicRegressor
    elif name == "KAN":
        from pycse.sklearn.kan import KAN

        return KAN
    elif name == "KANLLPR":
        from pycse.sklearn.kan_llpr import KANLLPR

        return KANLLPR
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
    elif name == "SISSO":
        from pycse.sklearn.sisso import SISSO

        return SISSO
    elif name == "SISSOEnsemble":
        from pycse.sklearn.sisso import SISSOEnsemble

        return SISSOEnsemble
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
