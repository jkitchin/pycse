"""K-fold Neural Network for Uncertainty Quantification.

This module provides a K-fold neural network model using JAX/Flax for uncertainty
quantification (UQ). The key idea is to train multiple neurons in the final layer
on different folds (subsets) of the training data, creating an ensemble of predictors
that naturally provides uncertainty estimates.

Key Concepts
------------
Traditional neural networks produce point predictions without uncertainty estimates.
The K-fold NN addresses this by:

1. **Architecture**: The network has shared hidden layers, but the final layer has
   multiple output neurons (one per fold).

2. **Training**: Each output neuron is trained on a different subset of the data
   (controlled by the `xtrain` parameter). The hidden layers see all data, but each
   output neuron specializes on its fold.

3. **Inference**: At prediction time, all output neurons make predictions. The mean
   gives the final prediction, and the standard deviation provides uncertainty.

The main hyperparameter affecting uncertainty is `xtrain` (fraction of data per fold):
- **Small xtrain (e.g., 0.1)**: Each neuron sees only 10% of data → diverse predictions
  → wider uncertainty estimates
- **Large xtrain (e.g., 1.0)**: Each neuron sees all data → similar predictions
  → narrow uncertainty (converges to standard NN)

Use Cases
---------
- Regression problems requiring uncertainty quantification
- Detecting extrapolation (uncertainty increases outside training domain)
- Active learning (sample where uncertainty is high)
- Safe decision-making (avoid high-uncertainty regions)

Classes
-------
KfoldNN
    sklearn-compatible k-fold neural network regressor with uncertainty quantification.

Notes
-----
- Requires JAX, Flax, and optax
- Uses L-BFGS optimizer for training
- Compatible with sklearn pipelines and cross-validation
- Uncertainty estimates are approximate (not Bayesian posteriors)

Examples
--------
Basic usage with 1D regression:

>>> import jax
>>> import numpy as np
>>> from pycse.sklearn.kfoldnn import KfoldNN
>>>
>>> # Generate noisy data
>>> key = jax.random.PRNGKey(42)
>>> x = np.linspace(0, 1, 100)[:, None]
>>> y = x**(1/3) + 0.1 * jax.random.normal(key, x.shape)
>>>
>>> # Create and fit model
>>> model = KfoldNN(layers=(1, 15, 25), xtrain=0.1)
>>> model.fit(x, y)
>>>
>>> # Predict with uncertainty
>>> x_test = np.array([[0.5]])
>>> mean, std = model.predict(x_test, return_std=True)
>>> print(f"Prediction: {mean[0]:.3f} ± {2*std[0]:.3f}")

Visualization:

>>> import matplotlib.pyplot as plt
>>> fig = model.plot(x, y, distribution=True)
>>> plt.show()

References
----------
This implementation is inspired by ensemble methods and dropout-based uncertainty
quantification, but uses explicit data folding in the final layer.
"""

import os
import jax

from jax import jit
import jax.numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from flax import linen as nn

from pycse.sklearn.optimizers import run_optimizer

# Enable 64-bit precision for better numerical stability
os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update("jax_enable_x64", True)


class _NN(nn.Module):
    """Internal Flax neural network module.

    This is a simple feedforward network with swish activation functions
    in hidden layers and a linear final layer. The final layer has multiple
    output neurons for the k-fold ensemble.

    Attributes
    ----------
    layers : tuple of int
        Number of neurons in each layer. For example, (5, 10, 25) creates:
        - Input layer: 5 features
        - Hidden layer: 10 neurons with swish activation
        - Output layer: 25 neurons (linear, for k-fold predictions)

    Notes
    -----
    This class is internal and not meant to be instantiated directly by users.
    Use the KfoldNN wrapper instead.
    """

    layers: tuple

    @nn.compact
    def __call__(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input features of shape (n_samples, n_features).

        Returns
        -------
        jax.numpy.ndarray
            Output predictions of shape (n_samples, n_outputs) where
            n_outputs = layers[-1].
        """
        # Hidden layers with swish activation
        for n_neurons in self.layers[:-1]:
            x = nn.Dense(n_neurons)(x)
            x = nn.swish(x)

        # Linear output layer (no activation)
        x = nn.Dense(self.layers[-1])(x)
        return x


class KfoldNN(BaseEstimator, RegressorMixin):
    """K-fold Neural Network for regression with uncertainty quantification.

    This model trains an ensemble of neural networks by dividing the training data
    into k folds and training each output neuron on a different fold. The ensemble
    predictions provide natural uncertainty estimates without requiring Bayesian
    inference or dropout.

    Parameters
    ----------
    layers : tuple of int
        Architecture specification. Each integer is the number of neurons in that
        layer. Example: (5, 20, 30) means 5 input features, 20 hidden neurons,
        and 30 output neurons (k=30 fold ensemble).
        Must be non-empty with positive integers only.
    xtrain : float, default=0.1
        Fraction of data each output neuron sees during training (0 < xtrain ≤ 1.0).
        Smaller values create more diversity → wider uncertainty bands.
        Larger values → predictions converge → narrower uncertainty.
        Typical values: 0.1 (diverse) to 0.3 (moderate).
    seed : int, default=19
        Random seed for reproducibility (affects data fold permutation).

    Attributes
    ----------
    nn : _NN
        Internal Flax neural network module.
    key : jax.random.PRNGKey
        Random key for JAX operations.
    optpars : dict, optional
        Optimized network parameters after fitting. Only available after fit().
    state : OptStep, optional
        Optimization state from LBFGS solver. Only available after fit().

    Methods
    -------
    fit(X, y, **kwargs)
        Train the k-fold neural network.
    predict(X, return_std=False)
        Make predictions with optional uncertainty estimates.
    score(X, y)
        Return R² score.
    plot(X, y, distribution=False)
        Visualize predictions and uncertainty.
    report()
        Print training statistics.

    Examples
    --------
    Train on synthetic data and visualize uncertainty:

    >>> import jax
    >>> import numpy as np
    >>> from pycse.sklearn.kfoldnn import KfoldNN
    >>>
    >>> # Generate data with noise
    >>> key = jax.random.PRNGKey(42)
    >>> x = np.linspace(0, 10, 100)[:, None]
    >>> y_true = np.sin(x)
    >>> y = y_true + 0.1 * jax.random.normal(key, x.shape)
    >>>
    >>> # Train model with 25 output neurons
    >>> model = KfoldNN(layers=(1, 20, 25), xtrain=0.15, seed=42)
    >>> model.fit(x, y)
    >>>
    >>> # Predict with uncertainty
    >>> x_new = np.array([[3.5], [5.0], [7.2]])
    >>> mean, std = model.predict(x_new, return_std=True)
    >>> for i, xi in enumerate(x_new):
    ...     print(f"x={xi[0]:.1f}: {mean[i]:.3f} ± {2*std[i]:.3f}")

    Compare different xtrain values:

    >>> models = {}
    >>> for xt in [0.1, 0.3, 1.0]:
    ...     model = KfoldNN(layers=(1, 15, 20), xtrain=xt)
    ...     model.fit(x, y)
    ...     models[xt] = model
    >>> # models[0.1] has widest uncertainty, models[1.0] has narrowest

    Notes
    -----
    - The hidden layers see all training data; only the final layer is split into folds
    - Training uses LBFGS optimizer (default: maxiter=1500, tol=1e-3)
    - Predictions are computed as the mean across output neurons
    - Uncertainty is estimated as the standard deviation across output neurons
    - Works best with smooth regression problems
    - For classification, consider other uncertainty quantification methods

    See Also
    --------
    sklearn.ensemble.BaggingRegressor : Another ensemble approach
    sklearn.neural_network.MLPRegressor : Standard neural network without UQ
    """

    def __init__(self, layers, xtrain=0.1, seed=19):
        """Initialize the K-fold neural network.

        Parameters
        ----------
        layers : tuple of int
            Network architecture. Example: (5, 20, 30) for 5 inputs,
            20 hidden neurons, 30 output neurons.
        xtrain : float, default=0.1
            Fraction of training data per fold (0 < xtrain ≤ 1.0).
        seed : int, default=19
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If layers is empty, contains non-positive integers, or is not a tuple.
        ValueError
            If xtrain is not in range (0, 1].
        TypeError
            If seed is not an integer.

        Examples
        --------
        >>> model = KfoldNN(layers=(5, 15, 20), xtrain=0.1)
        >>> print(model)
        K-fold Neural Network (not fitted)
          Architecture: (5, 15, 20)
          Training fraction: 0.10
          Output neurons: 20
        """
        # Validate layers
        if not isinstance(layers, tuple):
            raise TypeError(f"layers must be a tuple, got {type(layers).__name__}")
        if len(layers) == 0:
            raise ValueError("layers cannot be empty")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise ValueError(f"All layer sizes must be positive integers. Got: {layers}")

        # Validate xtrain
        if not isinstance(xtrain, (int, float)):
            raise TypeError(f"xtrain must be a number, got {type(xtrain).__name__}")
        if not (0 < xtrain <= 1.0):
            raise ValueError(f"xtrain must be in range (0, 1], got {xtrain}")

        # Validate seed
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

        self.layers = layers
        self.key = jax.random.PRNGKey(seed)
        self.nn = _NN(layers)
        self.xtrain = xtrain

    @property
    def is_fitted(self):
        """Check if the model has been fitted.

        Returns
        -------
        bool
            True if fit() has been called successfully, False otherwise.
        """
        return hasattr(self, "optpars")

    def fit(self, X, y, **kwargs):
        """Train the k-fold neural network on data.

        This method trains the network by minimizing the mean absolute error (MAE)
        using the LBFGS optimizer. Each output neuron is trained on a different
        fold of the data, determined by the xtrain parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features. Will be converted to JAX array.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values. Will be converted to JAX array and flattened.
        **kwargs : dict, optional
            Additional arguments passed to the L-BFGS optimizer:
            - maxiter : int, default=1500
                Maximum number of optimization iterations.
            - tol : float, default=1e-3
                Tolerance for convergence.

        Returns
        -------
        self
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If X is not 2D or y is not 1D/2D with compatible shape.
        ValueError
            If X and y have incompatible lengths.

        Notes
        -----
        - Can be called multiple times for retraining (warm start from previous parameters)
        - Uses smooth differentiable masking to select fold data
        - Minimizes sum of MAE across all folds
        - Training is deterministic given the same seed

        Examples
        --------
        Basic fitting:

        >>> import numpy as np
        >>> from pycse.sklearn.kfoldnn import KfoldNN
        >>>
        >>> X = np.random.randn(100, 3)
        >>> y = X @ np.array([1, 2, 3]) + 0.1 * np.random.randn(100)
        >>>
        >>> model = KfoldNN(layers=(3, 10, 15))
        >>> model.fit(X, y)
        >>> print(model.is_fitted)
        True

        Custom solver parameters:

        >>> model = KfoldNN(layers=(3, 10, 15))
        >>> model.fit(X, y, maxiter=3000, tol=1e-5)
        >>> model.report()
        """
        # Convert to JAX arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Validate X shape
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array, got {X.ndim}D array with shape {X.shape}. "
                "For 1D input, use X.reshape(-1, 1)"
            )

        # Validate and reshape y
        if y.ndim == 1:
            y = y[:, None]
        elif y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(f"y must be 1D or 2D with single column, got shape {y.shape}")
        else:
            raise ValueError(f"y must be 1D or 2D, got {y.ndim}D array with shape {y.shape}")

        # Validate compatible shapes
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got X: {len(X)}, y: {len(y)}")

        # Initialize or reuse parameters (allows retraining)
        if not self.is_fitted:
            params = self.nn.init(self.key, X)
        else:
            params = self.optpars

        # Determine last layer name and number of folds
        last_layer = f"Dense_{len(self.layers) - 1}"
        w = params["params"][last_layer]["kernel"].shape
        N = w[-1]  # number of output neurons (folds)

        # Create random permutations for each fold
        # Each fold gets a different ordering of the data indices
        folds = jax.random.permutation(
            self.key, np.tile(np.arange(0, len(X))[:, None], N), axis=0, independent=True
        ).T

        # Create smooth, differentiable cutoff function for fold masking
        # This ensures gradient flow during optimization
        fx = np.arange(0, len(X))
        _y = len(X) / 2 * (fx - len(X) * self.xtrain)
        fy = 1 - 0.5 * (np.tanh(_y / 2) + 1)  # Smooth step from 1 to 0

        @jit
        def objective(pars):
            """Compute total MAE across all folds."""
            total_error = 0

            for i, fold in enumerate(folds):
                # Predict for this fold's data ordering
                P = self.nn.apply(pars, np.asarray(X)[fold])

                # Extract errors for this fold's output neuron
                # fy masks out data we don't want this neuron to see
                errs = (P - y[fold])[:, i] * fy

                # Compute MAE for this fold
                mae = np.mean(np.abs(errs))
                total_error += mae

            return total_error

        # Set default solver parameters
        maxiter = kwargs.pop("maxiter", 1500)
        tol = kwargs.pop("tol", 1e-3)

        # Store maxiter for convergence checking
        self.maxiter = maxiter

        # Run L-BFGS optimization using optax
        self.optpars, self.state = run_optimizer(
            "lbfgs", objective, params, maxiter=maxiter, tol=tol, **kwargs
        )

        return self

    def predict(self, X, return_std=False):
        """Make predictions with optional uncertainty estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for prediction. Will be converted to 2D if needed.
        return_std : bool, default=False
            If True, return standard deviation (uncertainty) for each prediction.

        Returns
        -------
        predictions : jax.numpy.ndarray of shape (n_samples,)
            Mean predictions across all output neurons.
        std : jax.numpy.ndarray of shape (n_samples,), optional
            Standard deviation of predictions (only if return_std=True).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> import numpy as np
        >>> from pycse.sklearn.kfoldnn import KfoldNN
        >>>
        >>> # Assume model is already fitted
        >>> X_test = np.array([[1.5], [2.0], [3.5]])
        >>>
        >>> # Just predictions
        >>> y_pred = model.predict(X_test)
        >>>
        >>> # Predictions with uncertainty
        >>> y_pred, y_std = model.predict(X_test, return_std=True)
        >>> print(f"Prediction: {y_pred[0]:.3f} ± {2*y_std[0]:.3f}")

        Notes
        -----
        - Automatically handles 1D input by converting to 2D
        - Prediction is mean across all k output neurons
        - Uncertainty is standard deviation across outputs
        - Higher uncertainty indicates less confident predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")

        X = np.atleast_2d(X)
        P = self.nn.apply(self.optpars, X)

        if return_std:
            return np.mean(P, axis=1), np.std(P, axis=1)
        else:
            return np.mean(P, axis=1)

    def __call__(self, X, return_std=False, distribution=False):
        """Execute the model for predictions (alternate interface).

        This provides a more flexible prediction interface compared to predict(),
        allowing access to the full distribution of predictions from all output neurons.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for prediction.
        return_std : bool, default=False
            If True, return standard deviation for each prediction.
        distribution : bool, default=False
            If True, return full distribution (all k predictions per sample)
            instead of just the mean. Shape will be (n_samples, k).

        Returns
        -------
        predictions : jax.numpy.ndarray
            If distribution=False: shape (n_samples,) with mean predictions.
            If distribution=True: shape (n_samples, k) with all predictions.
        std : jax.numpy.ndarray of shape (n_samples,), optional
            Standard deviation (only if return_std=True).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> import numpy as np
        >>>
        >>> # Mean prediction (same as predict)
        >>> y_mean = model(X_test)
        >>>
        >>> # Full distribution of predictions
        >>> y_dist = model(X_test, distribution=True)
        >>> print(f"Shape: {y_dist.shape}")  # (n_samples, k)
        >>>
        >>> # Distribution with uncertainty
        >>> y_dist, y_std = model(X_test, distribution=True, return_std=True)

        Notes
        -----
        The distribution shows predictions from all k output neurons, useful for:
        - Visualizing ensemble diversity
        - Computing custom statistics (median, quantiles, etc.)
        - Understanding prediction uncertainty sources
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling. Call fit() first.")

        # Get all predictions from all output neurons
        P = self.nn.apply(self.optpars, X)
        se = P.std(axis=1)

        if not distribution:
            P = P.mean(axis=1)

        if return_std:
            return (P, se)
        else:
            return P

    def report(self):
        """Print and return training statistics.

        Returns
        -------
        dict or None
            If fitted, returns dictionary with training info:
            - 'iterations': number of optimization iterations
            - 'final_loss': final objective value
            - 'converged': whether optimization converged
            If not fitted, returns None and prints message.

        Examples
        --------
        >>> model.fit(X, y)
        >>> info = model.report()
        Iterations: 342, Loss: 0.1234
        >>> print(info['iterations'])
        342
        """
        if not self.is_fitted:
            print("Model not fitted yet. Call fit() first.")
            return None

        print(f"Iterations: {self.state.iter_num}, Loss: {self.state.value:.6f}")

        return {
            "iterations": int(self.state.iter_num),
            "final_loss": float(self.state.value),
            "converged": bool(self.state.iter_num < self.maxiter),
        }

    def plot(self, X, y, distribution=False):
        """Create visualization of predictions with uncertainty bands.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features (typically training or test data).
        y : array-like of shape (n_samples,)
            True target values.
        distribution : bool, default=False
            If True, plot all individual predictions from each output neuron
            (shows ensemble diversity). If False, only show mean ± 2std bands.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.

        Raises
        ------
        RuntimeError
            If model has not been fitted yet.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Basic plot with confidence bands
        >>> fig = model.plot(X_train, y_train)
        >>> plt.show()
        >>>
        >>> # Plot showing full distribution
        >>> fig = model.plot(X_train, y_train, distribution=True)
        >>> plt.title("K-fold NN with Ensemble Distribution")
        >>> plt.show()

        Notes
        -----
        The plot includes:
        - Blue points: actual data
        - Solid line: mean prediction
        - Dashed lines: ±2 standard deviations (~95% interval)
        - Faint lines (if distribution=True): individual neuron predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before plotting. Call fit() first.")

        P = self.nn.apply(self.optpars, X)
        mp = P.mean(axis=1)
        se = P.std(axis=1)

        plt.plot(X, y, "b.", label="data")
        plt.plot(X, mp, label="mean")
        plt.plot(X, mp + 2 * se, "k--")
        plt.plot(X, mp - 2 * se, "k--", label="+/- 2std")

        if distribution:
            plt.plot(X, P, alpha=0.2)

        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        return plt.gcf()

    def __repr__(self):
        """Return detailed string representation."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        n_outputs = self.layers[-1]

        repr_str = (
            f"KfoldNN(layers={self.layers}, xtrain={self.xtrain}, "
            f"outputs={n_outputs}, {fitted_str})"
        )

        if self.is_fitted:
            repr_str += f", loss={self.state.value:.6f}"

        return repr_str

    def __str__(self):
        """Return readable string description."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        n_outputs = self.layers[-1]
        n_hidden = len(self.layers) - 1

        desc = (
            f"K-fold Neural Network ({fitted_str})\n"
            f"  Architecture: {self.layers}\n"
            f"  Hidden layers: {n_hidden}\n"
            f"  Output neurons: {n_outputs}\n"
            f"  Training fraction: {self.xtrain:.2f}"
        )

        if self.is_fitted:
            desc += f"\n  Iterations: {self.state.iter_num}\n  Final loss: {self.state.value:.6f}"

        return desc
