"""
ZENN: Zentropy-Enhanced Neural Network

A thermodynamics-inspired computational framework for heterogeneous data-driven modeling.

This package provides sklearn-compatible estimators that implement zentropy theory
for classification and regression tasks, with support for:
- Multi-source heterogeneous data integration
- Energy landscape reconstruction
- Critical point detection
- High-order derivative computation

Reference:
    Wang, S.; Shang, S.-L.; Liu, Z.-K.; Hao, W. (2026).
    "ZENN: A thermodynamics-inspired computational framework for
    heterogeneous data-driven modeling." PNAS 123(1): e2511227122.

Upstream: https://github.com/WilliamMoriaty/ZENN (MIT License).
Vendored into pycse.sklearn — see NOTICE in this directory for attribution.
"""

from pycse.sklearn.zenn.estimators import ZENNClassifier, ZENNRegressor
from pycse.sklearn.zenn.estimators.regressor_nll import ZENNRegressorNLL

__version__ = "0.1.0"
__all__ = ["ZENNClassifier", "ZENNRegressor", "ZENNRegressorNLL"]
