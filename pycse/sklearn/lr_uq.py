"""
A Linear regressor with uncertainty quantification.
"""
import numpy as np
from pycse import regress
from pycse import predict as _predict
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegressionUQ(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.xtrain = np.array(X)
        self.ytrain = np.array(y)
        self.coefs_, self.pars_cint, self.pars_se = regress(X, y, rcond=None)
        return self

    def predict(self, X, return_std=False):
        y, _, se = _predict(self.xtrain, self.ytrain, self.coefs_, X)
        if return_std:
            return y, se
        else:
            return y
