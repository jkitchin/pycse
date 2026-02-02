"""SISSO (Sure Independence Screening and Sparsifying Operator) sklearn wrapper.

This module provides an sklearn-compatible wrapper around TorchSISSO for symbolic
regression with uncertainty quantification.

TorchSISSO discovers interpretable analytical expressions from data using the
SISSO algorithm. This wrapper makes it compatible with sklearn's estimator API.

Example usage:

    import numpy as np
    from pycse.sklearn import SISSO

    # Generate data from y = 2*x0 + x0*x1
    X = np.random.rand(100, 2)
    y = 2 * X[:, 0] + X[:, 0] * X[:, 1] + 0.1 * np.random.randn(100)

    # Fit SISSO
    model = SISSO(
        operators=['+', '-', '*', '/'],
        n_expansion=2,
        n_term=2,
        feature_names=['x0', 'x1']
    )
    model.fit(X, y)

    print(f"Discovered equation: {model.equation_}")

    # Predict with uncertainty
    y_pred, y_std = model.predict(X, return_std=True)

References:
    - TorchSISSO Paper: https://arxiv.org/abs/2410.01752
    - TorchSISSO GitHub: https://github.com/PaulsonLab/TorchSISSO

Requires: TorchSisso>=0.1.8
    pip install TorchSisso
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class SISSO(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for TorchSISSO symbolic regression with UQ.

    SISSO (Sure Independence Screening and Sparsifying Operator) discovers
    interpretable analytical expressions from data. This wrapper provides
    sklearn compatibility and adds calibrated uncertainty quantification
    using the hat matrix method.

    The final SISSO model is a linear combination of nonlinear features:
        y = c₀ + c₁·f₁(x) + c₂·f₂(x) + ... + cₜ·fₜ(x)

    Since this is linear in the feature space, we can compute prediction
    uncertainties using standard linear regression theory via the hat matrix.

    Parameters
    ----------
    operators : list, default=['+', '-', '*', '/']
        Operators for feature construction. Options include:
        - Arithmetic: '+', '-', '*', '/'
        - Mathematical: 'exp', 'ln', 'sqrt', 'abs', 'sin', 'cos'
    n_expansion : int, default=2
        Feature expansion depth. Higher values allow more complex expressions
        but increase computation time exponentially.
    n_term : int, default=2
        Number of terms in final equation. Typical values are 1-3.
    k : int, default=20
        Number of features retained per SIS iteration. Higher values explore
        more candidates but increase computation.
    use_gpu : bool, default=False
        Use GPU acceleration if available.
    feature_names : list, optional
        Names for input features. If None, uses x0, x1, x2, etc.
        Providing meaningful names makes equations more readable.

    Attributes
    ----------
    equation_ : str
        The discovered symbolic equation as a string.
    rmse_ : float
        Root mean squared error on training data.
    r2_ : float
        R² score on training data.
    sigma_ : float
        Estimated residual standard deviation.
    calibration_factor_ : float
        Calibration factor for uncertainty estimates.
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> import numpy as np
    >>> from pycse.sklearn import SISSO
    >>> X = np.random.rand(100, 2)
    >>> y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(100)
    >>> model = SISSO(n_expansion=1, n_term=1).fit(X, y)
    >>> print(model.equation_)  # doctest: +SKIP
    y = 0.5 + 0.99*x0 + 1.01*x1
    >>> y_pred, y_std = model.predict(X, return_std=True)

    Notes
    -----
    The uncertainty quantification uses the hat matrix method:

    1. Build design matrix Φ from the selected nonlinear features
    2. Compute leverage h_ii (diagonal of hat matrix H = Φ(ΦᵀΦ)⁻¹Φᵀ)
    3. Compute LOOCV residuals: e_i / (1 - h_ii) for proper out-of-sample calibration
    4. Estimate σ from LOOCV residuals (PRESS-like statistic)
    5. Prediction variance: Var(ŷ*) = σ² · (1 + φ*ᵀ(ΦᵀΦ)⁻¹φ*)
    6. Calibrate using LOOCV z-scores

    Using LOOCV residuals instead of training residuals provides proper
    out-of-sample calibration without requiring a separate validation set.
    This is valid because SISSO's final model is linear in its selected
    feature space, allowing efficient LOOCV computation via the hat matrix.

    **Limitation:** The hat matrix method detects extrapolation (unusual
    feature values) but may not increase uncertainty in input-space gaps
    where feature values remain within the training range. For example,
    with polynomial features [x, x², x³], a gap in x-space may not show
    increased uncertainty if the polynomial feature values still fall
    within the training range. For gap-aware uncertainty, consider
    Gaussian Processes or ensemble methods like DPOSE.
    """

    def __init__(
        self,
        operators=None,
        n_expansion=2,
        n_term=2,
        k=20,
        use_gpu=False,
        feature_names=None,
    ):
        self.operators = operators
        self.n_expansion = n_expansion
        self.n_term = n_term
        self.k = k
        self.use_gpu = use_gpu
        self.feature_names = feature_names

    def fit(self, X, y):
        """Fit SISSO model to discover symbolic equation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SISSO
            Returns self for method chaining.
        """
        try:
            from TorchSisso import SissoModel
        except ImportError:
            raise ImportError(
                "TorchSisso is required for SISSO. Install it with: pip install TorchSisso"
            )

        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Store training data for UQ computation
        self._X_train = X
        self._y_train = y

        # Create feature names if not provided
        if self.feature_names is None:
            self._feature_names = [f"x{i}" for i in range(X.shape[1])]
        else:
            self._feature_names = list(self.feature_names)

        # Build DataFrame (TorchSISSO requirement)
        df = pd.DataFrame(X, columns=self._feature_names)
        df.insert(0, "y", y)  # Target in first column

        # Default operators
        ops = self.operators if self.operators is not None else ["+", "-", "*", "/"]

        # Create and fit TorchSISSO model
        self._model = SissoModel(
            data=df,
            operators=ops,
            n_expansion=self.n_expansion,
            n_term=self.n_term,
            k=self.k,
            use_gpu=self.use_gpu,
        )

        result = self._model.fit()

        # Handle different return formats from TorchSISSO
        if len(result) >= 3:
            rmse, equation, r2 = result[0], result[1], result[2]
        else:
            rmse, equation = result[0], result[1]
            r2 = None

        # Store results
        self.equation_ = equation
        self.rmse_ = rmse
        self.r2_ = r2 if r2 is not None else self._compute_r2(X, y)

        # Parse equation to extract terms for UQ
        self._terms, self._coefficients, self._intercept = self._parse_equation(self.equation_)

        # Compute UQ parameters using hat matrix method
        self._compute_uq_params(X, y)

        self.is_fitted_ = True
        return self

    def _compute_r2(self, X, y):
        """Compute R² score if not provided by TorchSISSO."""
        y_pred = self._evaluate(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _parse_equation(self, equation):
        """Parse equation string to extract terms and coefficients.

        Parameters
        ----------
        equation : str
            Equation like "y = 0.5 + 1.2*x0 + -0.3*(x0*x1)"

        Returns
        -------
        terms : list of str
            Feature expressions (e.g., ['x0', '(x0*x1)'])
        coefficients : list of float
            Corresponding coefficients
        intercept : float
            The intercept term
        """
        # Remove "y = " or "y=" prefix
        eq = re.sub(r"^y\s*=\s*", "", equation.strip())

        terms = []
        coefficients = []
        intercept = 0.0

        # Pattern to match: optional sign, coefficient, optional *, term
        # This handles forms like: "0.5", "1.2*x0", "-0.3*(x0*x1)", "+2.5*exp(x0)"
        pattern = r"([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*?\s*([a-zA-Z_\(\)][^\s+-]*)?(?=\s*[+-]|$)"

        for match in re.finditer(pattern, eq):
            coef_str = match.group(1).replace(" ", "")
            term = match.group(2)

            if coef_str:
                try:
                    coef = float(coef_str)
                except ValueError:
                    continue

                if term and term.strip():
                    terms.append(term.strip())
                    coefficients.append(coef)
                else:
                    # This is the intercept
                    intercept += coef

        return terms, coefficients, intercept

    def _evaluate_term(self, term, X):
        """Evaluate a single term expression on X.

        Parameters
        ----------
        term : str
            Expression like 'x0', '(x0*x1)', 'exp(x0)'
        X : ndarray

        Returns
        -------
        values : ndarray of shape (n_samples,)
        """
        df = pd.DataFrame(X, columns=self._feature_names)

        # Build evaluation context with feature values
        context = {name: df[name].values for name in self._feature_names}

        # Add mathematical functions
        context.update(
            {
                "exp": np.exp,
                "ln": np.log,
                "log": np.log,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "pow": np.power,
            }
        )

        try:
            result = eval(term, {"__builtins__": {}}, context)
            return np.asarray(result).ravel()
        except Exception:
            # If evaluation fails, return zeros
            return np.zeros(X.shape[0])

    def _build_design_matrix(self, X):
        """Build design matrix from selected features.

        Parses the equation and evaluates each term to construct Φ.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        Phi : ndarray of shape (n_samples, n_terms + 1)
            Design matrix including intercept column.
        """
        n = X.shape[0]

        # Build Φ = [1, f1(X), f2(X), ...]
        Phi = np.ones((n, 1))  # Intercept column

        for term in self._terms:
            term_values = self._evaluate_term(term, X)
            Phi = np.column_stack([Phi, term_values])

        return Phi

    def _compute_uq_params(self, X, y):
        """Compute parameters for uncertainty quantification.

        Uses Leave-One-Out Cross-Validation (LOOCV) residuals for calibration,
        which can be computed efficiently for linear models using the hat matrix:

            LOOCV_residual_i = residual_i / (1 - h_ii)

        where h_ii is the leverage (diagonal of hat matrix). This provides
        proper out-of-sample calibration without actually refitting the model.
        """
        # Get predictions on training data
        y_pred = self._evaluate(X)
        n = len(y)

        # Build design matrix Φ from selected features
        Phi = self._build_design_matrix(X)
        p = Phi.shape[1]  # number of parameters (intercept + terms)

        # Compute (ΦᵀΦ)⁻¹ for hat matrix
        try:
            self._PhiTPhi_inv = np.linalg.pinv(Phi.T @ Phi)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            self._PhiTPhi_inv = np.eye(p) * 1e-6

        # Compute leverage (diagonal of hat matrix H = Φ(ΦᵀΦ)⁻¹Φᵀ)
        leverage = np.sum((Phi @ self._PhiTPhi_inv) * Phi, axis=1)
        # Clip leverage to avoid division issues (h_ii should be in [0, 1])
        leverage = np.clip(leverage, 0, 0.999)

        # Training residuals
        residuals = y - y_pred

        # LOOCV residuals: e_i / (1 - h_ii)
        # This is the residual we'd get if point i was left out during training
        loocv_residuals = residuals / (1 - leverage)

        # Estimate sigma using LOOCV residuals (PRESS statistic)
        # This gives a less biased estimate than training RSS
        self.sigma_ = np.sqrt(np.sum(loocv_residuals**2) / n)

        # Calibration using LOOCV z-scores
        # Prediction std for new points (not in training)
        std_pred = self.sigma_ * np.sqrt(1 + leverage)

        # Avoid division by zero
        std_pred = np.maximum(std_pred, 1e-10)

        # Z-scores using LOOCV residuals (proper out-of-sample calibration)
        z_scores = loocv_residuals / std_pred

        # Calibration factor (should be ~1 for well-calibrated)
        self.calibration_factor_ = np.std(z_scores) if len(z_scores) > 1 else 1.0
        if self.calibration_factor_ < 0.1:
            self.calibration_factor_ = 1.0  # Fallback

    def _evaluate(self, X):
        """Evaluate equation on X using TorchSISSO's evaluate method."""
        df = pd.DataFrame(X, columns=self._feature_names)
        result = self._model.evaluate(self.equation_, df)
        # TorchSISSO's evaluate returns (predictions, equation) tuple
        if isinstance(result, tuple):
            y_pred = result[0]
        else:
            y_pred = result
        return np.asarray(y_pred).ravel()

    def predict(self, X, return_std=False):
        """Predict using discovered equation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        return_std : bool, default=False
            If True, return calibrated standard deviation of predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        y_std : ndarray of shape (n_samples,), optional
            Calibrated standard deviation (if return_std=True).
        """
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        y_pred = self._evaluate(X)

        if return_std:
            # Compute prediction uncertainty using hat matrix
            Phi = self._build_design_matrix(X)
            leverage = np.sum((Phi @ self._PhiTPhi_inv) * Phi, axis=1)
            y_std = self.sigma_ * np.sqrt(1 + np.maximum(leverage, 0)) * self.calibration_factor_
            return y_pred, y_std

        return y_pred

    def score(self, X, y):
        """Return R² score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        score : float
            R² score.
        """
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def __repr__(self):
        """Return string representation."""
        params = [
            f"operators={self.operators}",
            f"n_expansion={self.n_expansion}",
            f"n_term={self.n_term}",
            f"k={self.k}",
        ]
        return f"SISSO({', '.join(params)})"
