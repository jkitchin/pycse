"""pyroxy - a surrogate decorator

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError


class _Surrogate:
    def __init__(self, func, model, tol=1, verbose=False):
        """Initialize a Surrogate function.

        Parameters
        ----------

        func : Callable
        Function that takes one argument

        model : sklearn model
        The model must be able to return std errors.

        tol : float optional, default=1
        Tolerance to use the surrogate.

        verbose : Boolean optional, default=False
        If truthy, output is more verbose.

        Returns
        -------
        return
        """

        self.func = func
        self.model = model
        self.tol = tol
        self.verbose = verbose
        self.xtrain = None
        self.ytrain = None

        self.ntrain = 0
        self.surrogate = 0

    def __call__(self, X):
        """Try to use the surrogate to predict X. if the predicted error is
        larger than self.tol, use the true function and retrain the surrogate.

        """
        try:
            pf, se = self.model.predict(X, return_std=True)

            # if we think it is accurate enough we return it
            if np.all(se < self.tol):
                self.surrogate += 1
                return pf
            else:
                if self.verbose:
                    print(
                        f"For {X} -> {pf} err={se} is greater than {self.tol},",
                        "running true function and returning function values and retraining",
                    )
                # Get the true value(s)
                y = self.func(X)

                # add it to the data. For now we add all the points
                self.xtrain = np.concatenate([self.xtrain, X], axis=0)
                self.ytrain = np.concatenate([self.ytrain, y])

                self.model.fit(self.xtrain, self.ytrain)
                self.ntrain += 1
                pf, se = self.model.predict(X, return_std=True)
                return y

        except NotFittedError:
            if self.verbose:
                print(f"Running {X} to initialize the model.")
            y = self.func(X)

            self.xtrain = X
            self.ytrain = y

            self.model.fit(X, y)
            self.ntrain += 1
            return y

    def plot(self):
        """Generate a parity plot of the surrogate.
        Shows approximate 95% uncertainty interval in shaded area.
        """
        yp, se = self.model.predict(self.xtrain, return_std=True)
        p = plt.plot(self.ytrain, yp, "b.")
        plt.fill_between(
            self.ytrain.flatten(),
            yp.flatten() + 2 * se.flatten(),
            yp.flatten() - 2 * se.flatten(),
            alpha=0.2,
        )
        plt.xlabel("Known y-values")
        plt.ylabel("Predicted y-values")
        plt.title(f"R$^2$ = {self.model.score(self.xtrain, self.ytrain)}")
        return p

    def __str__(self):
        """Returns a string representation of the surrogate."""
        return f"""{len(self.xtrain)} data points obtained.
        The model was fitted {self.ntrain} times.
        The surrogate was successful {self.surrogate} times."""


def Surrogate(function=None, *, model=None, tol=1, verbose=False):
    """Function Wrapper for _Surrogate class

    This allows me to use the class decorator with arguments.

    """

    def wrapper(function):
        return _Surrogate(function, model, tol, verbose)

    return wrapper
