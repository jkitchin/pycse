"""Leaf Models in sklearn.

This model is based on a DecisionTreeRegressor. When you train the model, it
first uses a DecisionTreeRegressor to divide the data set into leaves, then fits
your model to each leaf. You can request uncertainty that is computed from the
model fitted on each leaf.

This is not as rigorous as linear decision tree models, but it is conceptually
simple. I consider it a proof of concept model.

Features:
- Piecewise modeling: fits a separate model to each leaf
- Uncertainty quantification: supports models with return_std or uses residuals
- Calibration: post-hoc uncertainty scaling using validation data
- Diagnostics: plot(), report(), uncertainty_metrics()

Limitations:
- Tree splits are based on MSE, not leaf model performance (non-optimal)
- Extrapolation outside training bounds may be unreliable
- Requires sufficient samples per leaf for complex leaf models

Example:

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from pycse.sklearn.leaf_model import LeafModelRegressor

R = 8.314
k0, Ea = 6.79049544e+06, 4.02891385e+04

T = np.linspace(300, 600, 40)
k = k0 * np.exp(-Ea / R / T)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('poly', PolynomialFeatures(degree=2)),
                 ('Br', BayesianRidge())])

lt = LeafModelRegressor(leaf_model=pipe, min_samples_leaf=5)
lt.fit(T[:, None], k)

# With calibration
T_train, T_val = T[:30], T[30:]
k_train, k_val = k[:30], k[30:]
lt.fit(T_train[:, None], k_train, val_X=T_val[:, None], val_y=k_val)

f = np.linspace(200, 700)
pf, se = lt.predict(f[:, None], return_std=True)

# Diagnostics
lt.report()
lt.plot(T[:, None], k)
metrics = lt.uncertainty_metrics(T[:, None], k)

import matplotlib.pyplot as plt
plt.plot(T, k, '.')
plt.plot(f, pf.squeeze())
plt.plot(f, pf.squeeze() + se, f, pf.squeeze() - se)

"""

import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import clone


class LeafModelRegressor(DecisionTreeRegressor):
    """An sklearn Leaf Model class."""

    def __init__(self, leaf_model, **kwargs):
        """Initialize a LeafModel.

        LEAF_MODEL is an sklearn estimator.
        """
        self.leaf_model = leaf_model
        super().__init__(**kwargs)

    def fit(self, X, y, val_X=None, val_y=None):
        """Fit the model.

        First we fit the decision tree. Then we fit the leaves in the tree to
        the leaf_model. Each leaf gets its own model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        val_X : array-like, shape (n_val_samples, n_features), optional
            Validation data for calibration
        val_y : array-like, shape (n_val_samples,), optional
            Validation targets for calibration

        Returns
        -------
        self : object
            Returns self

        Notes
        -----
        This is not an optimal fit. The decision tree splits are based on MSE,
        not the leaf model's performance. It works reasonably well for many
        applications.
        """
        # Store input bounds for extrapolation detection
        self.X_min_ = np.min(X, axis=0)
        self.X_max_ = np.max(X, axis=0)

        # Fit the decision tree
        super().fit(X, y)

        # Now train the leaf models
        leaves = self.apply(X)
        self.leaf_models = {}
        self.leaf_stats_ = {}

        for leaf in set(leaves):
            # Get the x,y-points for this leaf
            ind = leaves == leaf
            _X = X[ind]
            _y = y[ind]

            # Store statistics for this leaf (for UQ fallback and diagnostics)
            self.leaf_stats_[leaf] = {
                "n_samples": len(_y),
                "X_mean": np.mean(_X, axis=0),
                "X_std": np.std(_X, axis=0),
                "y_mean": np.mean(_y),
                "y_std": np.std(_y),
            }

            # Check if leaf has enough samples
            if len(_y) < 2:
                warnings.warn(
                    f"Leaf {leaf} has only {len(_y)} sample(s). "
                    "This may cause fitting issues with complex models.",
                    UserWarning,
                )

            # Fit the model for this leaf
            self.leaf_models[leaf] = clone(self.leaf_model)
            try:
                self.leaf_models[leaf].fit(_X, _y)

                # Store residuals for fallback UQ
                y_pred = self.leaf_models[leaf].predict(_X)
                residuals = _y - y_pred
                self.leaf_stats_[leaf]["residual_std"] = np.std(residuals)
            except Exception as e:
                warnings.warn(
                    f"Failed to fit leaf model for leaf {leaf}: {e}. " "Using mean prediction.",
                    UserWarning,
                )
                # Fallback: store mean as a simple predictor
                self.leaf_models[leaf] = None

        # Calibration if validation data provided
        self.calibration_factor_ = 1.0
        if val_X is not None and val_y is not None:
            self.calibration_factor_ = self._calibrate(val_X, val_y)

        return self

    def predict(self, X, return_std=False, warn_extrapolation=True):
        """Predict values for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
        return_std : bool, default=False
            If True, return (predictions, std_errors)
        warn_extrapolation : bool, default=True
            If True, warn when predicting outside training data bounds

        Returns
        -------
        predictions : array, shape (n_samples,)
            Predicted values
        std_errors : array, shape (n_samples,), optional
            Standard errors (only if return_std=True)
        """
        # Check for extrapolation
        if warn_extrapolation and hasattr(self, "X_min_"):
            outside = np.any(X < self.X_min_, axis=1) | np.any(X > self.X_max_, axis=1)
            if np.any(outside):
                n_outside = np.sum(outside)
                warnings.warn(
                    f"{n_outside}/{len(X)} predictions are outside training data bounds. "
                    "Extrapolation may be unreliable.",
                    UserWarning,
                )

        # Get leaves for X that we are predicting
        pleaves = self.apply(X)

        predictions = np.zeros(X.shape[0])
        errors = np.zeros(X.shape[0])

        for leaf in set(pleaves):
            model = self.leaf_models[leaf]
            ind = pleaves == leaf

            # Handle case where leaf model failed to fit
            if model is None:
                predictions[ind] = self.leaf_stats_[leaf]["y_mean"]
                errors[ind] = self.leaf_stats_[leaf]["y_std"]
                continue

            try:
                # Try to get uncertainties from the model
                if return_std:
                    py, pse = model.predict(X[ind], return_std=True)
                else:
                    py = model.predict(X[ind])
                    pse = None

            except (ValueError, TypeError, AttributeError):
                # Model doesn't support return_std, use residual-based fallback
                py = model.predict(X[ind])
                if return_std:
                    # Use residual standard deviation from training
                    pse = np.full(py.shape, self.leaf_stats_[leaf].get("residual_std", np.nan))
                else:
                    pse = None

            predictions[ind] = py
            if return_std and pse is not None:
                errors[ind] = pse

        # Apply calibration factor
        if return_std and hasattr(self, "calibration_factor_"):
            errors = errors * self.calibration_factor_

        if return_std:
            return np.array(predictions), np.array(errors)
        else:
            return np.array(predictions)

    def _calibrate(self, val_X, val_y):
        """Calibrate uncertainties using validation data.

        Parameters
        ----------
        val_X : array-like
            Validation features
        val_y : array-like
            Validation targets

        Returns
        -------
        calibration_factor : float
            Factor to scale uncertainties
        """
        y_pred, y_std = self.predict(val_X, return_std=True, warn_extrapolation=False)

        # Calculate z-scores
        z_scores = (val_y - y_pred) / (y_std + 1e-10)

        # Ideal: std(z_scores) = 1.0
        # If < 1: uncertainties overestimate, if > 1: uncertainties underestimate
        std_z = np.std(z_scores)

        if std_z < 0.1:
            warnings.warn(
                "Calibration factor is very small. Uncertainties may be unreliable.", UserWarning
            )

        return std_z

    def score(self, X, y):
        """Return the R² score.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values

        Returns
        -------
        score : float
            R² score
        """
        y_pred = self.predict(X, warn_extrapolation=False)
        return r2_score(y, y_pred)

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        # Get parent class params (DecisionTreeRegressor)
        params = super().get_params(deep=False)

        # Add leaf_model param
        params["leaf_model"] = self.leaf_model

        # If deep, add nested leaf_model params
        if deep and hasattr(self.leaf_model, "get_params"):
            leaf_params = self.leaf_model.get_params(deep=True)
            for key, val in leaf_params.items():
                params[f"leaf_model__{key}"] = val

        return params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        # Separate leaf_model params from tree params
        leaf_model_params = {}
        other_params = {}

        for key, value in params.items():
            if key == "leaf_model":
                other_params[key] = value
            elif key.startswith("leaf_model__"):
                # Remove prefix for leaf_model
                leaf_key = key[len("leaf_model__") :]
                leaf_model_params[leaf_key] = value
            else:
                other_params[key] = value

        # Set tree params
        super().set_params(**other_params)

        # Set leaf_model params
        if leaf_model_params and hasattr(self.leaf_model, "set_params"):
            self.leaf_model.set_params(**leaf_model_params)

        return self

    def report(self):
        """Print a summary report of the model.

        Shows tree structure, leaf statistics, and model information.
        """
        print("=" * 70)
        print("LeafModelRegressor Summary")
        print("=" * 70)

        # Tree structure info
        n_leaves = self.get_n_leaves()
        depth = self.get_depth()
        print("\nTree Structure:")
        print(f"  Number of leaves: {n_leaves}")
        print(f"  Maximum depth: {depth}")

        if hasattr(self, "calibration_factor_"):
            print(f"  Calibration factor: {self.calibration_factor_:.4f}")

        # Leaf model info
        print(f"\nLeaf Model: {type(self.leaf_model).__name__}")

        # Leaf statistics
        if hasattr(self, "leaf_stats_"):
            print("\nLeaf Statistics:")
            print(
                f"  {'Leaf':<8} {'N Samples':<12} {'Y Mean':<12} {'Y Std':<12} {'Residual Std':<15}"
            )
            print("  " + "-" * 63)

            for leaf_id in sorted(self.leaf_stats_.keys()):
                stats = self.leaf_stats_[leaf_id]
                n_samples = stats["n_samples"]
                y_mean = stats["y_mean"]
                y_std = stats["y_std"]
                res_std = stats.get("residual_std", np.nan)

                print(
                    f"  {leaf_id:<8} {n_samples:<12} {y_mean:<12.4f} "
                    f"{y_std:<12.4f} {res_std:<15.4f}"
                )

        print("=" * 70)

    def plot(self, X, y, title="LeafModelRegressor Predictions"):
        """Visualize predictions with uncertainties.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data (only 1D supported for visualization)
        y : array-like, shape (n_samples,)
            True values
        title : str, optional
            Plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting")
            return

        if X.shape[1] != 1:
            print("Plotting only supported for 1D input data")
            return

        X_plot = X.ravel()
        y_pred, y_std = self.predict(X, return_std=True, warn_extrapolation=False)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Predictions with uncertainties
        ax = axes[0]
        ax.scatter(X_plot, y, alpha=0.5, label="True data", s=20)
        ax.plot(X_plot, y_pred, "r-", label="Predictions", linewidth=2)
        ax.fill_between(
            X_plot, y_pred - y_std, y_pred + y_std, alpha=0.3, label="±1 std", color="red"
        )
        ax.fill_between(
            X_plot,
            y_pred - 2 * y_std,
            y_pred + 2 * y_std,
            alpha=0.2,
            label="±2 std",
            color="red",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Residuals
        ax = axes[1]
        residuals = y - y_pred
        ax.scatter(X_plot, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.fill_between(X_plot, -y_std, y_std, alpha=0.3, color="gray", label="±1 std")
        ax.set_xlabel("X")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def uncertainty_metrics(self, X, y):
        """Compute uncertainty quantification metrics.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples,)
            True values

        Returns
        -------
        metrics : dict
            Dictionary containing various uncertainty metrics
        """
        y_pred, y_std = self.predict(X, return_std=True, warn_extrapolation=False)

        # Basic prediction metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Uncertainty calibration metrics
        residuals = y - y_pred
        z_scores = residuals / (y_std + 1e-10)

        # Check if uncertainties are well-calibrated
        # Ideal: mean(|z|) ≈ 0.798 (for standard normal), std(z) ≈ 1.0
        mean_abs_z = np.mean(np.abs(z_scores))
        std_z = np.std(z_scores)

        # Negative log-likelihood (assuming Gaussian)
        nll = 0.5 * np.mean(np.log(2 * np.pi * y_std**2) + z_scores**2)

        # Miscalibration area (ideal = 0)
        # Fraction of points within k*std for k=1,2,3
        within_1std = np.mean(np.abs(z_scores) <= 1.0)  # Should be ~0.68
        within_2std = np.mean(np.abs(z_scores) <= 2.0)  # Should be ~0.95
        within_3std = np.mean(np.abs(z_scores) <= 3.0)  # Should be ~0.997

        miscalibration = (
            abs(within_1std - 0.6827) + abs(within_2std - 0.9545) + abs(within_3std - 0.9973)
        ) / 3.0

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mean_abs_z_score": mean_abs_z,
            "std_z_score": std_z,
            "nll": nll,
            "within_1std": within_1std,
            "within_2std": within_2std,
            "within_3std": within_3std,
            "miscalibration_area": miscalibration,
            "mean_uncertainty": np.mean(y_std),
            "std_uncertainty": np.std(y_std),
        }

        return metrics

    def print_metrics(self, X, y):
        """Print formatted uncertainty metrics.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples,)
            True values
        """
        metrics = self.uncertainty_metrics(X, y)

        print("\n" + "=" * 70)
        print("Uncertainty Quantification Metrics")
        print("=" * 70)

        print("\nPrediction Quality:")
        print(f"  R² Score:              {metrics['r2']:.4f}")
        print(f"  RMSE:                  {metrics['rmse']:.4f}")
        print(f"  MAE:                   {metrics['mae']:.4f}")

        print("\nUncertainty Calibration:")
        print(f"  Mean |Z-score|:        {metrics['mean_abs_z_score']:.4f} (ideal: 0.798)")
        print(f"  Std Z-score:           {metrics['std_z_score']:.4f} (ideal: 1.000)")
        print(f"  NLL:                   {metrics['nll']:.4f}")
        print(f"  Miscalibration Area:   {metrics['miscalibration_area']:.4f} (ideal: 0.000)")

        print("\nCoverage:")
        print(f"  Within 1σ:             {metrics['within_1std']:.1%} (ideal: 68.3%)")
        print(f"  Within 2σ:             {metrics['within_2std']:.1%} (ideal: 95.4%)")
        print(f"  Within 3σ:             {metrics['within_3std']:.1%} (ideal: 99.7%)")

        print("\nUncertainty Statistics:")
        print(f"  Mean uncertainty:      {metrics['mean_uncertainty']:.4f}")
        print(f"  Std uncertainty:       {metrics['std_uncertainty']:.4f}")

        print("=" * 70 + "\n")
