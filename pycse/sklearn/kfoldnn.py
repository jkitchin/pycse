"""A K-fold Neural network model in jax.

The idea of the k-fold model is that you train each neuron in the last layer on
a different fold of data. Then, at inference time you get a distribution of
predictions that you can use for uncertainty quantification.

The main hyperparameter that affects the distribution is the fraction of data
used. Empirically I find that a fraction of 0.1 works pretty well. Note that the
neurons before the last layer all end up seeing all the data, it is only the
last layer that sees different parts of the data. If you use a fraction of 1.0,
then each neuron converges to the same result.

There isn't currently an obvious way to choose a fraction that leads to the
"right" UQ distribution. You can try many values and see what works best.

Example usage:

import jax
import numpy as np
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(19)

x = np.linspace(0, 1, 100)[:, None]
y = x**(1/3) + (1 + jax.random.normal(key, x.shape) * 0.05)


from pycse.sklearn.kfoldnn import KfoldNN
model = KfoldNN((1, 15, 25), xtrain=0.1)

model.fit(x, y)

model.report()
print(model.score(x, y))
model.plot(x, y, distribution=True);

"""

import os
import jax


from jax import jit
import jax.numpy as np
from jax import value_and_grad
from jaxopt import LBFGS
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from flax import linen as nn

os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


class _NN(nn.Module):
    """A flax neural network.

    layers: a Tuple of integers. Each integer is the number of neurons in that
    layer.
    """

    layers: tuple

    @nn.compact
    def __call__(self, x):
        for i in self.layers[0:-1]:
            x = nn.Dense(i)(x)
            x = nn.swish(x)

        # Linear last layer
        x = nn.Dense(self.layers[-1])(x)
        return x


class KfoldNN(BaseEstimator, RegressorMixin):
    """sklearn compatible model for a k-fold neural network."""

    def __init__(self, layers, xtrain=0.1, seed=19):
        """Initialize a k-fold nn.

        args:
            layers : tuple of integers for neurons in each layer
            xtrain: fraction of data to use in each fold.
        """
        self.layers = layers
        self.key = jax.random.PRNGKey(seed)
        self.nn = _NN(layers)
        self.xtrain = xtrain

    def fit(self, X, y, **kwargs):
        """Fit the kfold nn.

        Args:
            X : a 2d array of x values
            y : an array of y values.

            kwargs are passed to the LBGF solver.
        """
        # This allows retraining.
        if not hasattr(self, "optpars"):
            params = self.nn.init(self.key, X)
        else:
            params = self.optpars

        last_layer = f"Dense_{len(self.layers) - 1}"
        w = params["params"][last_layer]["kernel"].shape
        N = w[-1]  # number of functions in the last layer

        folds = jax.random.permutation(
            self.key, np.tile(np.arange(0, len(X))[:, None], N), axis=0, independent=True
        ).T

        # make a smooth, differentiable cutoff
        fx = np.arange(0, len(X))

        # We use fy to mask out the errors for the dataset we don't want
        _y = len(X) / 2 * (fx - len(X) * self.xtrain)
        fy = 1 - 0.5 * (np.tanh(_y / 2) + 1)

        @jit
        def objective(pars):
            agge = 0

            for i, fold in enumerate(folds):
                # predict for a fold
                P = self.nn.apply(pars, np.asarray(X)[fold])
                errs = (P - y[fold])[:, i] * fy  # errors for this fold

                mae = np.mean(np.abs(errs))  # MAE for the fold
                agge += mae
            return agge

        if "maxiter" not in kwargs:
            kwargs["maxiter"] = 1500

        if "tol" not in kwargs:
            kwargs["tol"] = 1e-3

        solver = LBFGS(fun=value_and_grad(objective), value_and_grad=True, **kwargs)

        self.optpars, self.state = solver.run(params)

    def report(self):
        """Print the state variables."""
        print(f"Iterations: {self.state.iter_num} Value: {self.state.value}")

    def predict(self, X, return_std=False):
        """Predict the model for X.

        Args:
            X: a 2d array of points to predict
            return_std: Boolean, if true, return error estimate for each point.

        Returns:
            if return_std is False, the predictions, else (predictions, errors)
        """
        X = np.atleast_2d(X)
        P = self.nn.apply(self.optpars, X)

        if return_std:
            return np.mean(P, axis=1), np.std(P, axis=1)
        else:
            return np.mean(P, axis=1)

    def __call__(self, X, return_std=False, distribution=False):
        """Execute the model.

        Args:
            X: a 2d array to make predictions for.
            return_std: Boolean, if true return errors for each point
            distribution: Boolean, if true return the distribution, else the mean.

        """
        if not hasattr(self, "optpars"):
            raise Exception("You need to fit the model first.")

        # get predictions
        P = self.nn.apply(self.optpars, X)
        se = P.std(axis=1)
        if not distribution:
            P = P.mean(axis=1)

        if return_std:
            return (P, se)
        else:
            return P

    def plot(self, X, y, distribution=False):
        """Return a plot.

        Args:
            X: 2d array of data
            y: corresponding y-values
            distribution: Boolean, if true, plot the distribution of predictions.
        """
        P = self.nn.apply(self.optpars, X)
        mp = P.mean(axis=1)
        se = P.std(axis=1)

        plt.plot(X, y, "b.", label="data")
        plt.plot(X, mp, label="mean")
        plt.plot(X, mp + 2 * se, "k--")
        plt.plot(X, mp - 2 * se, "k--", label="+/- 2sd")
        if distribution:
            plt.plot(X, P, alpha=0.2)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        return plt.gcf()
