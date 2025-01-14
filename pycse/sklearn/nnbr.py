"""Neural network with Bayesian Linear regression.

Use a neural network as a nonlinear feature generator, then use Bayesian Linear
regression for the last layer so you can also get UQ.

Small example:

nn = MLPRegressor((20, 200), activation='relu', solver='lbfgs', max_iter=1000)
br = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)

nnbr = NeuralNetworkBLR(nn, br)
nn.fit(x_train, y_train)

nnbr.fit(x_train, y_train)

x_fit = np.linspace(0.95, 3)

m, s = nnbr.predict(x_fit[:, None], return_std=True)
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network._base import ACTIVATIONS


class NeuralNetworkBLR(BaseEstimator, RegressorMixin):
    """sklearn-compatible neural network with Bayesian Regression in last layer.

    The idea is you fit a neural network and replace the last linear layer with
    a Bayesian linear regressor so you can estimate uncertainty.
    """

    def __init__(self, nn, br):
        """Initialize the Neural Network Bayesian Linear Regressor.

        nn: An sklearn.neural_network.MLPRegressor instance
        br: An sklearn.linear_model.BayesianRidge instance
        """
        self.nn = nn
        self.br = br

    def _feat(self, X):
        """Return neural network features for X."""
        weights = self.nn.coefs_
        biases = self.nn.intercepts_

        # Get the output of last hidden layer
        feat = X @ weights[0] + biases[0]
        ACTIVATIONS[self.nn.activation](feat)  # works in place
        for i in range(1, len(weights) - 1):
            feat = feat @ weights[i] + biases[i]
            ACTIVATIONS[self.nn.activation](feat)
        return feat

    def fit(self, X, y):
        """Fit the regressor to X, y.

        This first fits the NeuralNetwork instance. Then it gets the features
        from the output layer and uses those in the Bayesian linear regressor.
        """
        # initial fit
        self.nn.fit(X, y)

        # Bayesian linear regression
        self.br.fit(self._feat(X), y)

        return self

    def predict(self, X, return_std=False):
        """Predict output values for X.

        if return_std is truthy, also return the standard deviation for each
        prediction.

        """
        return self.br.predict(self._feat(X), return_std=return_std)
