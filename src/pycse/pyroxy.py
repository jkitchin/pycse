"""pyroxy - a surrogate decorator

TODO: What about train / test splits?

TODO: what about a random fraction of function values instead of surrogate?

This is proof of concept code and it is not obviously the best approach. A
notable limitation is that pickle and joblib cannot save this. It works ok with
dill so far.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
import dill


class MaxCallsExceededException(Exception):
    """Raised when maximum number of function calls is exceeded."""


class _Surrogate:
    def __init__(self, func, model, tol=1, max_calls=-1, verbose=False):
        """Initialize a Surrogate function.

        Parameters
        ----------

        func : Callable

          Function that takes one argument

        model : sklearn model

          The model must be able to return std errors.

        tol : float optional, default=1

          Tolerance to use the surrogate. If the predicted error is less than
          this we use the surrogate, otherwise use the true function and
          retrain.

        max_calls : int, default=-1,

          Maximum number of calls to allow. An exception is raised if you exceed
          this. -1 means no limit.

        verbose : Boolean optional, default=False
        If truthy, output is more verbose.

        """
        self.func = func
        self.model = model
        self.tol = tol
        self.max_calls = max_calls
        self.verbose = verbose
        self.xtrain = None
        self.ytrain = None

        self.ntrain = 0
        self.surrogate = 0
        self.func_calls = 0

    def add(self, X):
        """Get data for X, add it and retrain.
        Use this to bypass the logic for using the surrogate.
        """
        if (self.max_calls >= 0) and (self.func_calls + 1) > self.max_calls:
            raise MaxCallsExceededException(f"Max func calls ({self.max_calls}) will be exceeded")

        y = self.func(X)
        self.func_calls += 1

        # add it to the data. For now we add all the points
        if self.xtrain is not None:
            self.xtrain = np.concatenate([self.xtrain, X], axis=0)
            self.ytrain = np.concatenate([self.ytrain, y])
        else:
            self.xtrain = X
            self.ytrain = y

        self.model.fit(self.xtrain, self.ytrain)
        self.ntrain += 1
        return y

    def test(self, X):
        """Run a test on X.
        Runs true function on X, computes prediction errors.

        Returns:
        True if the actual errors are less than the tolerance.
        """
        # Ensure X is 2D for sklearn compatibility
        X = np.atleast_2d(X)

        if (self.max_calls >= 0) and (self.func_calls + 1) > self.max_calls:
            raise MaxCallsExceededException(f"Max func calls ({self.max_calls}) will be exceeded")

        y = self.func(X)
        self.func_calls += 1
        yp, ypse = self.model.predict(X, return_std=True)

        errs = y - yp

        if self.verbose:
            print(
                f"""Testing {X}
            y = {y}
            yp = {yp}

            ypse = {ypse}
            ypse < tol = {np.abs(ypse) < self.tol}

            errs = {errs}
            errs < tol = {np.abs(errs) < self.tol}
            """
            )
        return (np.max(ypse) < self.tol) and (np.max(np.abs(errs)) < self.tol)

    def __call__(self, X):
        """Try to use the surrogate to predict X. if the predicted error is
        larger than self.tol, use the true function and retrain the surrogate.

        """
        # Ensure X is 2D for sklearn compatibility
        X = np.atleast_2d(X)

        try:
            pf, se = self.model.predict(X, return_std=True)

            # if we think it is accurate enough we return it
            if np.all(se < self.tol):
                self.surrogate += 1
                return pf.flatten()
            else:
                if self.verbose:
                    print(
                        f"For {X} -> {pf} err={se} is greater than {self.tol},",
                        "running true function and returning function values and retraining",
                    )

                if (self.max_calls >= 0) and (self.func_calls + 1) > self.max_calls:
                    raise MaxCallsExceededException(
                        f"Max func calls ({self.max_calls}) will be exceeded"
                    )
                # Get the true value(s)
                y = self.func(X)
                self.func_calls += 1

                # add it to the data. For now we add all the points
                if self.xtrain is not None:
                    self.xtrain = np.concatenate([self.xtrain, X], axis=0)
                    self.ytrain = np.concatenate([self.ytrain, y])
                else:
                    # First data point
                    self.xtrain = X
                    self.ytrain = y

                self.model.fit(self.xtrain, self.ytrain)
                self.ntrain += 1
                return y

        except (AttributeError, NotFittedError):
            if self.verbose:
                print(f"Running {X} to initialize the model.")
            y = self.func(X)
            self.func_calls += 1

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

        # sort these so the points are plotted sequentially in order
        sind = np.argsort(self.ytrain.flatten())
        y = self.ytrain.flatten()[sind]
        yp = yp.flatten()[sind]
        se = se.flatten()[sind]

        p = plt.plot(y, yp, "b.")
        plt.fill_between(
            y,
            yp + 2 * se,
            yp - 2 * se,
            alpha=0.2,
        )
        plt.xlabel("Known y-values")
        plt.ylabel("Predicted y-values")
        plt.title(f"R$^2$ = {self.model.score(self.xtrain, self.ytrain)}")
        return p

    def __str__(self):
        """A string representation."""

        yp, ypse = self.model.predict(self.xtrain, return_std=True)

        errs = self.ytrain - yp

        return f"""{len(self.xtrain)} data points obtained.
        The model was fitted {self.ntrain} times.
        The surrogate was successful {self.surrogate} times.

        model score: {self.model.score(self.xtrain, self.ytrain)}
        Errors:
        MAE: {np.mean(np.abs(errs))}
        RMSE: {np.sqrt(np.mean(errs**2))}
        (tol = {self.tol})

        """

    def dump(self, fname="model.pkl"):
        """Save the current surrogate to fname."""
        with open(fname, "wb") as f:
            f.write(dill.dumps(self))

        return fname


def Surrogate(function=None, *, model=None, tol=1, verbose=False, max_calls=-1):
    """Function Wrapper for _Surrogate class

    This allows me to use the class decorator with arguments.

    """

    def wrapper(function):
        return _Surrogate(function, model=model, tol=tol, verbose=verbose, max_calls=max_calls)

    return wrapper


# This seems clunky, but I want this to have the syntax:
# Surrogate.load(fname)


def load(fname="model.pkl"):
    """Load a surrogate from fname."""
    with open(fname, "rb") as f:
        return dill.loads(f.read())


Surrogate.load = load


class ActiveSurrogate:
    """Build surrogate models using active learning.

    This class provides methods to automatically build surrogate models by
    iteratively sampling an input domain using acquisition functions to select
    informative points.
    """

    @classmethod
    def build(
        cls,
        func,
        bounds,
        model,
        acquisition="ei",
        stopping_criterion="mean_ratio",
        stopping_threshold=1.5,
        n_initial=None,
        batch_size=1,
        max_iterations=1000,
        n_test_points=None,
        n_candidates=None,
        verbose=False,
        callback=None,
        tol=1.0,
    ):
        """Build a surrogate model using active learning.

        Parameters
        ----------
        func : callable
            Function to surrogate. Must accept 2D array and return 1D array.

        bounds : list of tuples
            Domain bounds as [(low1, high1), (low2, high2), ...].

        model : sklearn model
            Model with predict(X, return_std=True) interface.

        acquisition : str, default='ei'
            Acquisition function: 'ei', 'ucb', 'pi', 'variance'.

        stopping_criterion : str, default='mean_ratio'
            Stopping criterion: 'mean_ratio', 'percentile', 'absolute', 'convergence'.

        stopping_threshold : float, default=1.5
            Threshold value for stopping criterion.

        n_initial : int, optional
            Initial samples. Defaults to max(10, 5*n_dims).

        batch_size : int, default=1
            Number of points to sample per iteration.

        max_iterations : int, default=1000
            Maximum iterations before stopping.

        n_test_points : int, optional
            Test points for uncertainty estimation. Defaults to 100*n_dims.

        n_candidates : int, optional
            Candidate points for acquisition. Defaults to 50*n_dims.

        verbose : bool, default=False
            Print progress information.

        callback : callable, optional
            Function called each iteration: callback(iteration, history).

        tol : float, default=1.0
            Tolerance for returned _Surrogate object.

        Returns
        -------
        surrogate : _Surrogate
            Fitted surrogate model.

        history : dict
            Training history with metrics per iteration.
        """
        # Placeholder implementation
        raise NotImplementedError("ActiveSurrogate.build() not yet implemented")
