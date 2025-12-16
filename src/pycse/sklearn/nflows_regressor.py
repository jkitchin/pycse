"""
Sklearn-compatible Normalizing Flow Regressor.

This module provides an sklearn-compatible wrapper around normalizing flows
for regression with both forward and inverse inference capabilities.

Example
-------
>>> from nflows_regressor import NFlowsRegressor
>>> import numpy as np
>>>
>>> # Generate data
>>> X = np.random.uniform(0, 1, (100, 1))
>>> y = np.sin(2 * np.pi * X).ravel() + 0.1 * np.random.randn(100)
>>>
>>> # Fit model
>>> model = NFlowsRegressor(num_layers=5, max_epochs=200)
>>> model.fit(X, y)
>>>
>>> # Forward prediction
>>> y_pred = model.predict(X)
>>>
>>> # Inverse inference: find X that produces y=0
>>> X_inverse = model.inverse(np.array([[0.0]]), n_samples=100)

Reference
---------
nflows: https://github.com/bayesiains/nflows
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    RandomPermutation,
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NFlowsRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible normalizing flow regressor with forward and inverse inference.

    This estimator uses Neural Spline Flows (NSF) to model conditional distributions,
    enabling both forward prediction p(y|X) and inverse inference p(X|y).

    The key insight is that we train two separate flows:
    1. A forward flow p(y|X) for prediction
    2. An inverse flow p(X|y) for inverse problems

    Parameters
    ----------
    num_layers : int, default=5
        Number of flow transformation layers. More layers increase expressivity
        but also training time and risk of overfitting.

    hidden_features : int, default=64
        Number of hidden units in each transformation layer.

    num_bins : int, default=8
        Number of bins for the rational quadratic spline. More bins allow
        more complex transformations but increase parameters.

    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.

    max_epochs : int, default=500
        Maximum number of training epochs.

    batch_size : int, default=128
        Batch size for training.

    patience : int, default=20
        Early stopping patience (epochs without improvement).

    verbose : bool, default=False
        Whether to print training progress.

    random_state : int, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features (set after fit).

    n_outputs_ : int
        Number of output dimensions (set after fit).

    forward_flow_ : Flow
        Trained forward flow p(y|X).

    inverse_flow_ : Flow
        Trained inverse flow p(X|y).

    forward_losses_ : list
        Training losses for forward flow.

    inverse_losses_ : list
        Training losses for inverse flow.

    Examples
    --------
    >>> import numpy as np
    >>> from nflows_regressor import NFlowsRegressor
    >>>
    >>> # 1D regression with heteroscedastic noise
    >>> X = np.random.uniform(0, 1, (500, 1))
    >>> noise = 0.1 + 0.2 * X
    >>> y = np.sin(2 * np.pi * X) + noise * np.random.randn(500, 1)
    >>> y = y.ravel()
    >>>
    >>> model = NFlowsRegressor(num_layers=5, max_epochs=300, verbose=True)
    >>> model.fit(X, y)
    >>>
    >>> # Forward prediction with uncertainty
    >>> y_pred, y_std = model.predict(X[:10], n_samples=100, return_std=True)
    >>>
    >>> # Inverse inference
    >>> X_inverse = model.inverse(np.array([[0.5]]), n_samples=200)

    Notes
    -----
    The flow uses Neural Spline Flows (NSF) with rational quadratic splines,
    which are flexible and numerically stable. The tails are set to 'linear'
    with a bound of 5.0 to handle values outside the training range.

    For best results:
    - Normalize your data (handled internally)
    - Use enough training data (>500 samples recommended)
    - Tune num_layers and hidden_features for your problem complexity

    References
    ----------
    .. [1] Durkan et al. "Neural Spline Flows" (2019)
           https://arxiv.org/abs/1906.04032
    """

    def __init__(
        self,
        num_layers=5,
        hidden_features=64,
        num_bins=8,
        learning_rate=1e-3,
        max_epochs=500,
        batch_size=128,
        patience=20,
        verbose=False,
        random_state=None,
    ):
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state

    def _build_flow(self, features, context_features):
        """Build a conditional Neural Spline Flow.

        Parameters
        ----------
        features : int
            Dimension of the variable being modeled.
        context_features : int
            Dimension of the conditioning variable.

        Returns
        -------
        flow : Flow
            The constructed normalizing flow.
        """
        transforms = []
        for _ in range(self.num_layers):
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=features,
                    hidden_features=self.hidden_features,
                    context_features=context_features,
                    num_bins=self.num_bins,
                    tails="linear",
                    tail_bound=5.0,
                )
            )
            if features > 1:
                transforms.append(RandomPermutation(features=features))

        return Flow(
            transform=CompositeTransform(transforms), distribution=StandardNormal([features])
        )

    def _train_flow(self, flow, X, y, desc="Training"):
        """Train a conditional flow.

        Parameters
        ----------
        flow : Flow
            The flow to train.
        X : ndarray
            Context/conditioning data.
        y : ndarray
            Target data to model.
        desc : str
            Description for logging.

        Returns
        -------
        losses : list
            Training losses per epoch.
        """
        flow.to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0
        losses = []

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = -flow.log_prob(batch_y, context=batch_X).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"{desc}: Early stopping at epoch {epoch + 1}")
                break

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"{desc} Epoch {epoch + 1}/{self.max_epochs}, Loss: {avg_loss:.4f}")

        return losses

    def fit(self, X, y):
        """Fit the forward and inverse flows.

        This trains two normalizing flows:
        1. Forward flow: p(y|X) for prediction
        2. Inverse flow: p(X|y) for inverse inference

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.atleast_2d(X)
        y = np.atleast_2d(y) if y.ndim == 1 else y
        if y.shape[0] != X.shape[0]:
            y = y.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        # Store data statistics for normalization
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0) + 1e-8
        self.y_mean_ = y.mean(axis=0)
        self.y_std_ = y.std(axis=0) + 1e-8

        X_norm = (X - self.X_mean_) / self.X_std_
        y_norm = (y - self.y_mean_) / self.y_std_

        # Build and train forward flow p(y|X)
        self.forward_flow_ = self._build_flow(
            features=self.n_outputs_, context_features=self.n_features_in_
        )
        self.forward_losses_ = self._train_flow(self.forward_flow_, X_norm, y_norm, "Forward flow")

        # Build and train inverse flow p(X|y)
        self.inverse_flow_ = self._build_flow(
            features=self.n_features_in_, context_features=self.n_outputs_
        )
        self.inverse_losses_ = self._train_flow(self.inverse_flow_, y_norm, X_norm, "Inverse flow")

        return self

    def predict(self, X, n_samples=1, return_std=False):
        """Forward inference: predict y given X.

        Samples from the learned distribution p(y|X) and returns the mean
        prediction. Optionally returns standard deviation for uncertainty.

        Parameters
        ----------
        X : array-like of shape (n_points, n_features)
            Input data.
        n_samples : int, default=1
            Number of samples to draw per input for averaging.
            Use n_samples > 1 for better mean estimates and uncertainty.
        return_std : bool, default=False
            If True, return standard deviation of samples.

        Returns
        -------
        y_pred : ndarray of shape (n_points,) or (n_points, n_outputs)
            Mean prediction across samples.
        y_std : ndarray of shape (n_points,) or (n_points, n_outputs)
            Standard deviation (only if return_std=True).
        """
        X = np.atleast_2d(X)
        X_norm = (X - self.X_mean_) / self.X_std_
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

        self.forward_flow_.eval()
        with torch.no_grad():
            # nflows returns shape: (n_points, n_samples, n_outputs)
            y_samples = self.forward_flow_.sample(n_samples, context=X_tensor)

        y_samples = y_samples.cpu().numpy()
        y_samples = y_samples * self.y_std_ + self.y_mean_

        # Mean over samples dimension (axis=1): (n_points, n_outputs)
        y_mean = y_samples.mean(axis=1)

        # For single output, return 1D array (n_points,)
        if self.n_outputs_ == 1:
            y_mean = y_mean.ravel()

        if return_std:
            y_std = y_samples.std(axis=1)
            if self.n_outputs_ == 1:
                y_std = y_std.ravel()
            return y_mean, y_std

        return y_mean

    def inverse(self, y, n_samples=100):
        """Inverse inference: sample X given y.

        Uses the inverse flow to sample from p(X|y), which is useful for:
        - Design optimization (finding inputs for desired outputs)
        - Root finding (what X gives y=target?)
        - Inverse problems in general

        Parameters
        ----------
        y : array-like of shape (n_targets,) or (n_queries, n_targets)
            Target values to invert.
        n_samples : int, default=100
            Number of samples to draw per target.

        Returns
        -------
        X_samples : ndarray of shape (n_samples, n_features) or
                    (n_queries, n_samples, n_features)
            Samples from p(X|y).

        Examples
        --------
        >>> # Find X values where the model predicts y=0
        >>> X_inverse = model.inverse(np.array([[0.0]]), n_samples=500)
        >>> print(f"Mean X: {X_inverse.mean():.3f}, Std: {X_inverse.std():.3f}")
        """
        y = np.atleast_2d(y)
        if y.shape[1] != self.n_outputs_:
            y = y.reshape(-1, self.n_outputs_)

        y_norm = (y - self.y_mean_) / self.y_std_
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).to(device)

        self.inverse_flow_.eval()
        with torch.no_grad():
            # nflows returns shape: (n_queries, n_samples, n_features)
            X_samples = self.inverse_flow_.sample(n_samples, context=y_tensor)

        X_samples = X_samples.cpu().numpy()
        X_samples = X_samples * self.X_std_ + self.X_mean_

        # For single query, return (n_samples, n_features)
        if y.shape[0] == 1:
            return X_samples[0]
        return X_samples

    def sample_posterior(self, X, n_samples=100):
        """Sample from the posterior p(y|X).

        This is useful for uncertainty quantification and visualizing
        the full predictive distribution.

        Parameters
        ----------
        X : array-like of shape (n_features,) or (n_queries, n_features)
            Input data.
        n_samples : int, default=100
            Number of posterior samples.

        Returns
        -------
        y_samples : ndarray of shape (n_samples, n_outputs) or
                    (n_queries, n_samples, n_outputs)
            Samples from the posterior.

        Examples
        --------
        >>> # Get posterior samples for uncertainty visualization
        >>> y_samples = model.sample_posterior(np.array([[0.5]]), n_samples=1000)
        >>> import matplotlib.pyplot as plt
        >>> plt.hist(y_samples.ravel(), bins=50, density=True)
        >>> plt.xlabel('y')
        >>> plt.title('Posterior p(y|X=0.5)')
        """
        X = np.atleast_2d(X)
        X_norm = (X - self.X_mean_) / self.X_std_
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

        self.forward_flow_.eval()
        with torch.no_grad():
            # nflows returns shape: (n_queries, n_samples, n_outputs)
            y_samples = self.forward_flow_.sample(n_samples, context=X_tensor)

        y_samples = y_samples.cpu().numpy()
        y_samples = y_samples * self.y_std_ + self.y_mean_

        # For single query, return (n_samples, n_outputs)
        if X.shape[0] == 1:
            return y_samples[0]
        return y_samples

    def log_prob(self, X, y):
        """Compute log probability log p(y|X).

        Useful for model comparison and likelihood-based analysis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        log_prob : ndarray of shape (n_samples,)
            Log probability for each sample.
        """
        X = np.atleast_2d(X)
        y = np.atleast_2d(y) if y.ndim == 1 else y
        if y.shape[0] != X.shape[0]:
            y = y.reshape(-1, 1)

        X_norm = (X - self.X_mean_) / self.X_std_
        y_norm = (y - self.y_mean_) / self.y_std_

        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).to(device)

        self.forward_flow_.eval()
        with torch.no_grad():
            log_prob = self.forward_flow_.log_prob(y_tensor, context=X_tensor)

        # Adjust for the Jacobian of the normalization transform
        log_prob = log_prob - np.sum(np.log(self.y_std_))

        return log_prob.cpu().numpy()

    def score(self, X, y):
        """Return the negative mean squared error (for sklearn compatibility).

        Higher is better (sklearn convention).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test input data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True target values.

        Returns
        -------
        score : float
            Negative mean squared error.
        """
        y_pred = self.predict(X, n_samples=50)
        return -mean_squared_error(y, y_pred)


# Convenience function for quick model creation
def create_flow_regressor(complexity="medium", **kwargs):
    """Create an NFlowsRegressor with preset configurations.

    Parameters
    ----------
    complexity : str, default='medium'
        Model complexity level: 'low', 'medium', or 'high'.
        - 'low': Fast training, suitable for simple problems
        - 'medium': Balanced, good default for most problems
        - 'high': More expressive, for complex relationships
    **kwargs
        Additional arguments passed to NFlowsRegressor.

    Returns
    -------
    model : NFlowsRegressor
        Configured regressor.

    Examples
    --------
    >>> model = create_flow_regressor('low', max_epochs=100)
    >>> model.fit(X, y)
    """
    presets = {
        "low": {"num_layers": 3, "hidden_features": 32, "num_bins": 4},
        "medium": {"num_layers": 5, "hidden_features": 64, "num_bins": 8},
        "high": {"num_layers": 8, "hidden_features": 128, "num_bins": 16},
    }

    if complexity not in presets:
        raise ValueError(f"complexity must be one of {list(presets.keys())}")

    config = presets[complexity].copy()
    config.update(kwargs)

    return NFlowsRegressor(**config)
