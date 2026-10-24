"""
A Linear regressor with uncertainty quantification.
"""

import numpy as np
from pycse import regress
from pycse import predict as _predict
from sklearn.base import BaseEstimator, RegressorMixin

from pycse.PYCSE import _hessian_shift
from pycse.sklearn.mip import Affine, MIPExportMixin, MIPNetwork, std_from_covariance


class LinearRegressionUQ(MIPExportMixin, BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.xtrain_ = np.array(X)
        self.ytrain_ = np.array(y)
        self.coefs_, self.pars_cint_, self.pars_se_ = regress(
            self.xtrain_, self.ytrain_, rcond=None
        )
        return self

    def predict(self, X, return_std=False):
        X = np.array(X)
        y, _, se = _predict(self.xtrain_, self.ytrain_, self.coefs_, X)
        if return_std:
            return y, se
        else:
            return y

    def mip_network(self):
        """Return the fitted model as a ``MIPNetwork`` (a single affine layer).

        The prediction ``X @ coefs_`` is linear, so the export needs no binaries. The
        std is the one ``predict(return_std=True)`` gives,
        ``sqrt(mse * (1 + x^T (X^T X + eps I)^{-1} x))``, a second-order cone.
        """
        if not hasattr(self, "coefs_"):
            raise ValueError("Fit the model before exporting it.")
        X, y = self.xtrain_, self.ytrain_
        coefs = np.asarray(self.coefs_, dtype=float)
        W = coefs.reshape(len(coefs), -1)
        layers = [Affine(W, np.zeros(W.shape[1]))]

        # Same estimates as pycse.predict: mse with n - p dof and a regularized inverse.
        n, npars = X.shape[0], W.shape[0]
        mse = np.atleast_1d(np.sum((y.reshape(n, -1) - X @ W) ** 2, axis=0) / (n - npars))
        hat = X.T @ X
        inv = np.linalg.pinv(hat + np.eye(npars) * _hessian_shift(hat, 1e-5, 1.05))
        std = [std_from_covariance(inv, -1, scale=np.sqrt(s2), d=1.0) for s2 in mse]
        return MIPNetwork(layers, std=std)
