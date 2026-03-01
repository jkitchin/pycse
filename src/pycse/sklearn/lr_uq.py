"""
A Linear regressor with uncertainty quantification.
"""

import numpy as np
from pycse import regress
from pycse import predict as _predict
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegressionUQ(BaseEstimator, RegressorMixin):
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
