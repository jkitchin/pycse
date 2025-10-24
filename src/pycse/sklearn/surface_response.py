"""Surface Response Methodology for Design of Experiments.

This module provides tools for creating and analyzing response surface designs,
which are commonly used in experimental design to model the relationship between
multiple input variables and one or more output responses.
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from pycse.sklearn.lr_uq import LinearRegressionUQ
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from pyDOE3 import fullfact, ff2n, pbdesign, gsd, bbdesign, ccdesign, lhs
import numpy as np
import pandas as pd
import tabulate


class SurfaceResponse(Pipeline):
    """A class for Surface Response Design of Experiments (DOE).

    This class combines experimental design generation with polynomial regression
    modeling to create response surface models. It supports various DOE types and
    can handle multiple input factors and output responses.

    Parameters
    ----------
    inputs : list of str
        Names of each input factor/variable
    outputs : list of str
        Names of each output response
    bounds : array-like of shape (n_inputs, 2), optional
        Bounds for each input factor. Each row is [xmin, xmax].
        The design assumes [-1, 1] map to these bounds.
    design : str, default='bbdesign'
        Type of experimental design. Options are:
        - 'fullfact': Full factorial design
        - 'ff2n': 2-level full factorial design
        - 'pbdesign': Plackett-Burman design
        - 'gsd': Generalized subset design
        - 'bbdesign': Box-Behnken design
        - 'ccdesign': Central composite design
        - 'lhs': Latin hypercube sampling
    model : sklearn estimator, optional
        Custom model to use. If None (default), uses a pipeline with
        MinMaxScaler, 2nd-order PolynomialFeatures, and LinearRegressionUQ.
    **kwargs
        Additional keyword arguments passed to the pyDOE3 design function.

    Attributes
    ----------
    inputs : list of str
        Input factor names
    outputs : list of str
        Output response names
    bounds : ndarray
        Bounds for each input factor
    input : DataFrame
        The generated design matrix
    output : DataFrame
        The measured/simulated outputs

    Examples
    --------
    >>> sr = SurfaceResponse(
    ...     inputs=['temperature', 'pressure'],
    ...     outputs=['yield'],
    ...     bounds=[[100, 200], [1, 5]],
    ...     design='bbdesign'
    ... )
    >>> design_df = sr.design()
    >>> # Run experiments and collect data
    >>> sr.set_output([[0.75], [0.82], [0.68], ...])
    >>> sr.fit()
    >>> print(sr.summary())
    """

    def __init__(
        self, inputs=None, outputs=None, bounds=None, design="bbdesign", model=None, **kwargs
    ):
        """Initialize a SurfaceResponse object."""
        # Input validation
        if inputs is None or not inputs:
            raise ValueError("inputs must be a non-empty list of factor names")
        if outputs is None or not outputs:
            raise ValueError("outputs must be a non-empty list of response names")
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list of strings")
        if not isinstance(outputs, list):
            raise TypeError("outputs must be a list of strings")

        self.inputs = inputs
        self.outputs = outputs

        # Validate and set bounds
        if bounds is not None:
            bounds = np.array(bounds)
            if bounds.shape != (len(inputs), 2):
                raise ValueError(f"bounds must have shape ({len(inputs)}, 2), got {bounds.shape}")
            if np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ValueError("All bounds must have min < max")
        self.bounds = bounds if bounds is None else np.array(bounds)

        # Generate design
        if design == "fullfact":
            self._design = fullfact(**kwargs)
        elif design == "ff2n":
            self._design = ff2n(len(inputs))
        elif design == "pbdesign":
            self._design = pbdesign(len(inputs))
        elif design == "gsd":
            self._design = gsd(**kwargs)
        elif design == "bbdesign":
            self._design = bbdesign(len(inputs), **kwargs)
        elif design == "ccdesign":
            self._design = ccdesign(len(inputs), **kwargs)  # FIXED: was ==
        elif design == "lhs":
            self._design = lhs(len(inputs), **kwargs)
        else:
            raise ValueError(
                f"Unsupported design option: {design}. "
                f"Choose from: fullfact, ff2n, pbdesign, gsd, bbdesign, ccdesign, lhs"
            )

        self.model = model

        # Initialize pipeline
        if model is None:
            self.default = True
            scaler = MinMaxScaler(feature_range=(-1, 1)).set_output(transform="pandas")
            super().__init__(
                steps=[
                    ("minmax", scaler),
                    ("poly", PolynomialFeatures(2)),
                    ("surface response", LinearRegressionUQ()),
                ]
            )
        else:
            self.default = False
            super().__init__(steps=[("usermodel", model)])

    def design(self, shuffle=True):
        """Create a design dataframe with experimental conditions.

        Generates the experimental design matrix by scaling the normalized
        design points to the actual factor bounds.

        Parameters
        ----------
        shuffle : bool, default=True
            If True, randomize the order of experimental runs.

        Returns
        -------
        DataFrame
            Design matrix with columns corresponding to input factors.
            Each row represents one experimental run.

        Examples
        --------
        >>> sr = SurfaceResponse(
        ...     inputs=['temp', 'press'],
        ...     outputs=['yield'],
        ...     bounds=[[100, 200], [1, 5]]
        ... )
        >>> design = sr.design(shuffle=True)
        """
        design = self._design.copy()
        nrows, ncols = design.shape

        # Scale from [-1, 1] to actual bounds
        # Formula: x = (Xsc - a) * (xmax - xmin) / (b - a) + xmin
        # where a=-1, b=1, Xsc is the scaled design value
        a, b = -1, 1

        if self.bounds is not None:
            mins = self.bounds[:, 0]
            maxs = self.bounds[:, 1]
            design = (design - a) * (maxs - mins) / (b - a) + mins

        df = pd.DataFrame(data=design, columns=self.inputs)

        if shuffle:
            df = df.sample(frac=1)

        self.input = df
        return df

    def set_output(self, data):
        """Set the output response data from experiments.

        Parameters
        ----------
        data : array-like of shape (n_experiments, n_outputs)
            Experimental results. Each row should correspond to the same
            row in the input design matrix.

        Returns
        -------
        DataFrame
            The output dataframe with columns corresponding to output responses.

        Raises
        ------
        ValueError
            If data shape doesn't match the design or outputs specification.
        AttributeError
            If design() has not been called yet.

        Examples
        --------
        >>> sr.design()
        >>> results = [[0.75], [0.82], [0.68]]  # From experiments
        >>> sr.set_output(results)
        """
        if not hasattr(self, "input"):
            raise AttributeError("Must call design() before set_output()")

        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if len(data) != len(self.input):
            raise ValueError(f"data has {len(data)} rows but design has {len(self.input)} rows")
        if data.shape[1] != len(self.outputs):
            raise ValueError(
                f"data has {data.shape[1]} columns but {len(self.outputs)} outputs expected"
            )

        index = self.input.index
        df = pd.DataFrame(data, index=index, columns=self.outputs)
        self.output = df
        return self.output

    def fit(self, X=None, y=None):
        """Fit the response surface model to the experimental data.

        Parameters
        ----------
        X : array-like, optional
            Input data. If None, uses self.input from design().
        y : array-like, optional
            Output data. If None, uses self.output from set_output().

        Returns
        -------
        self
            Fitted estimator.
        """
        if X is None and y is None:
            if not hasattr(self, "input") or not hasattr(self, "output"):
                raise AttributeError("Must set input and output data before fitting")
            X, y = self.input, self.output
        return super().fit(X, y)

    def score(self, X=None, y=None):
        """Compute the R² coefficient of determination.

        Parameters
        ----------
        X : array-like, optional
            Input data. If None, uses self.input.
        y : array-like, optional
            Output data. If None, uses self.output.

        Returns
        -------
        float
            R² score of the model.
        """
        if X is None and y is None:
            X, y = self.input, self.output
        return super().score(X, y)

    def parity(self):
        """Create a parity plot comparing true and predicted values.

        Returns
        -------
        Figure
            Matplotlib figure object containing the parity plot.
        """
        X, y = self.input, self.output
        pred = self.predict(X)

        plt.figure(figsize=(6, 6))
        plt.scatter(y, pred, alpha=0.6, edgecolors="k", linewidth=0.5)

        # Plot diagonal line
        min_val = min(y.min().min(), pred.min())
        max_val = max(y.max().max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect fit")

        plt.xlabel("True Value", fontsize=12)
        plt.ylabel("Predicted Value", fontsize=12)
        plt.title(f"Parity Plot (R² = {self.score():.3f})", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()

    def _sigfig(self, x, n=3):
        """Round x to n significant figures.

        Parameters
        ----------
        x : float
            Value to round
        n : int, default=3
            Number of significant figures

        Returns
        -------
        float
            Rounded value

        Notes
        -----
        Adapted from https://gist.github.com/ttamg/3f65227fd580b3d8dc8ba91e01507280
        """
        if x == 0:
            return 0
        return np.round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))

    def summary(self):
        """Generate a comprehensive summary of the fitted model.

        Includes overall metrics (R², MAE, RMSE) and for each feature:
        the parameter value, confidence interval, standard error, and
        statistical significance.

        Returns
        -------
        str
            Formatted summary string with model statistics and parameters.

        Notes
        -----
        Significance is determined by whether the 95% confidence interval
        contains zero. If it does not contain zero, the parameter is
        considered statistically significant.
        """
        X, y = self.input, self.output

        s = [f"{len(X)} data points"]
        yp = self.predict(X)
        errs = y - yp

        if self.default:
            features = self["poly"].get_feature_names_out()

            pars = self["surface response"].coefs_
            pars_cint = self["surface response"].pars_cint
            pars_se = self["surface response"].pars_se

            nrows, ncols = pars.shape

            mae = [self._sigfig(x) for x in (np.abs(errs).mean())]
            rmse = [self._sigfig(x) for x in np.sqrt((errs**2).mean())]

            s += [f"  R² score: {self.score(X, y):.4f}"]
            s += [
                f"  MAE  = {mae}",
                "",
                f"  RMSE = {rmse}",
                "",
            ]

            # Handle corner case for single output
            if ncols == 1:
                pars_cint = [pars_cint]

            for i in range(ncols):  # Loop over outputs
                data = []
                s += [f"Output {i}: {y.columns[i]}"]
                for j, name in enumerate(features):
                    # i is the ith output
                    # j is the jth feature
                    # cint has shape (n_outputs, n_features, 2)
                    # se has shape (n_features, n_outputs)
                    data += [
                        [
                            f"{name}_{i}",
                            pars[j][i],
                            pars_cint[i][j][0],
                            pars_cint[i][j][1],
                            pars_se[j][i],
                            np.sign(pars_cint[i][j][0] * pars_cint[i][j][1]) > 0,
                        ]
                    ]
                s += [
                    tabulate.tabulate(
                        data,
                        headers=[
                            "variable",
                            "value",
                            "ci_lower",
                            "ci_upper",
                            "std_err",
                            "significant",
                        ],
                        tablefmt="orgtbl",
                    )
                ]
                s += [""]
        else:
            s += ["User-defined model:", repr(self["usermodel"])]

            mae = [self._sigfig(x) for x in (np.abs(errs).mean())]
            rmse = [self._sigfig(x) for x in np.sqrt((errs**2).mean())]

            s += [f"  R² score: {self.score(X, y):.4f}"]
            s += [
                f"  MAE  = {mae}",
                "",
                f"  RMSE = {rmse}",
                "",
            ]

        return "\n".join(s)

    def __repr__(self):
        """Return a string representation of the SurfaceResponse object."""
        return (
            f"SurfaceResponse(inputs={self.inputs}, outputs={self.outputs}, "
            f"n_experiments={len(self.input) if hasattr(self, 'input') else 0})"
        )

    def __str__(self):
        """Return a readable string description."""
        fitted = hasattr(self, "input") and hasattr(self, "output")
        status = "fitted" if fitted else "not fitted"
        return (
            f"SurfaceResponse with {len(self.inputs)} inputs, "
            f"{len(self.outputs)} outputs ({status})"
        )
