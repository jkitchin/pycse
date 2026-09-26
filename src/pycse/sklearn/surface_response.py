"""Surface Response Methodology for Design of Experiments.

This module provides tools for creating and analyzing response surface designs,
which are commonly used in experimental design to model the relationship between
multiple input variables and one or more output responses.

Coded units
-----------
Every design is generated in *coded* units in which ``-1`` and ``+1`` are the
lower and upper ``bounds`` of each input. The generated experiments therefore
never leave the bounds you specify. The default regression model scales the
inputs back to these coded units before fitting, so the reported coefficients
are in coded units (the change in the response per half-range of each input).
"""

import warnings
import weakref

from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted
from pycse.sklearn.lr_uq import LinearRegressionUQ
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from pyDOE3 import fullfact, ff2n, pbdesign, gsd, bbdesign, ccdesign, lhs
import numpy as np
import pandas as pd
import tabulate

_DESIGNS = ("fullfact", "ff2n", "pbdesign", "gsd", "bbdesign", "ccdesign", "lhs")


class CodedScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Scale inputs to coded units in which the bounds map to [-1, 1].

    Parameters
    ----------
    bounds : array-like of shape (n_features, 2), optional
        ``[xmin, xmax]`` for each feature. ``xmin`` maps to -1 and ``xmax`` to +1.
        If None, the minimum and maximum of the training data are used instead
        (equivalent to ``MinMaxScaler(feature_range=(-1, 1))``).
    """

    def __init__(self, bounds=None):
        """Initialize the scaler."""
        self.bounds = bounds

    def fit(self, X, y=None):
        """Compute the center and half-range of each feature."""
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        self.n_features_in_ = Xa.shape[1]

        if self.bounds is None:
            lo, hi = Xa.min(axis=0), Xa.max(axis=0)
        else:
            b = np.asarray(self.bounds, dtype=float)
            if b.shape != (self.n_features_in_, 2):
                raise ValueError(
                    f"bounds must have shape ({self.n_features_in_}, 2), got {b.shape}"
                )
            lo, hi = b[:, 0], b[:, 1]

        half_range = (hi - lo) / 2.0
        half_range[half_range == 0] = 1.0  # constant feature: avoid division by zero
        self.center_ = (hi + lo) / 2.0
        self.half_range_ = half_range
        return self

    def transform(self, X):
        """Map X to coded units."""
        check_is_fitted(self, "center_")
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        return (Xa - self.center_) / self.half_range_

    def inverse_transform(self, X):
        """Map coded units back to physical units."""
        check_is_fitted(self, "center_")
        return np.asarray(X, dtype=float) * self.half_range_ + self.center_


class _DesignName(str):
    """The ``design`` constructor parameter (a str) that is also callable.

    ``SurfaceResponse.design`` has to hold the constructor argument so that
    sklearn's ``get_params``/``clone`` work. Older code called ``sr.design()`` to
    generate the experiments; calling this string forwards to
    ``SurfaceResponse.generate_design`` with a DeprecationWarning.
    """

    def __new__(cls, value):
        obj = super().__new__(cls, value)
        obj._owner = None
        return obj

    def __call__(self, *args, **kwargs):
        owner = self._owner() if self._owner is not None else None
        if owner is None:
            raise TypeError("This design name is not attached to a SurfaceResponse.")
        warnings.warn(
            "Calling SurfaceResponse.design() is deprecated; use generate_design() instead. "
            "The `design` attribute now holds the design name passed to the constructor.",
            DeprecationWarning,
            stacklevel=2,
        )
        return owner.generate_design(*args, **kwargs)

    def __reduce__(self):
        return (_DesignName, (str(self),))

    def __copy__(self):
        return _DesignName(str(self))

    def __deepcopy__(self, memo):
        return _DesignName(str(self))


class _DesignParam:
    """Descriptor storing the ``design`` parameter and binding it to its owner."""

    def __set_name__(self, owner, name):
        self.key = "_" + name + "_param"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.key)
        if isinstance(value, _DesignName):
            value._owner = weakref.ref(obj)
        return value

    def __set__(self, obj, value):
        if isinstance(value, str):
            bound = value._owner() if isinstance(value, _DesignName) and value._owner else None
            if not isinstance(value, _DesignName) or (bound is not None and bound is not obj):
                value = _DesignName(value)
        obj.__dict__[self.key] = value


def _index_to_coded(design, levels):
    """Map level indices 0..L-1 (fullfact, gsd) to coded values in [-1, 1]."""
    design = np.asarray(design, dtype=float)
    levels = np.asarray(levels, dtype=float)
    span = np.where(levels > 1, levels - 1, 1.0)
    coded = 2.0 * design / span - 1.0
    coded[:, levels <= 1] = 0.0
    return coded


def _coded_design(design, k, kwargs):
    """Generate a design for k factors in coded units, where [-1, 1] are the bounds."""
    kwargs = dict(kwargs)
    if design in ("fullfact", "gsd"):
        levels = kwargs.get("levels")
        if levels is None:
            raise ValueError(f"design='{design}' requires the `levels` keyword argument")
        if len(levels) != k:
            raise ValueError(f"levels must have one entry per input ({k}), got {len(levels)}")
        raw = fullfact(**kwargs) if design == "fullfact" else gsd(**kwargs)
        coded = _index_to_coded(raw, levels)
    elif design == "ff2n":
        coded = ff2n(k)
    elif design == "pbdesign":
        coded = pbdesign(k)
    elif design == "bbdesign":
        if k >= 3:
            coded = bbdesign(k, **kwargs)
        else:
            # Box-Behnken needs at least 3 factors. Use the closest 3-level design that
            # stays inside the bounds: a face-centered central composite design.
            n_center = kwargs.get("center", 3)
            warnings.warn(
                f"Box-Behnken designs require at least 3 inputs; using a face-centered "
                f"central composite design (3 levels, {n_center} center points) for {k} "
                f"input(s) instead.",
                UserWarning,
                stacklevel=4,
            )
            if k == 1:
                coded = np.array([[-1.0], [1.0]] + [[0.0]] * n_center)
            else:
                coded = np.vstack([ccdesign(k, center=(0, 0), face="ccf"), np.zeros((n_center, k))])
    elif design == "ccdesign":
        # Face-centered (alpha = 1) by default so every run is within the bounds.
        kwargs.setdefault("face", "ccf")
        coded = ccdesign(k, **kwargs)
    elif design == "lhs":
        # pyDOE3 lhs samples the unit hypercube [0, 1]^k
        coded = 2.0 * np.asarray(lhs(k, **kwargs), dtype=float) - 1.0
    else:
        raise ValueError(f"Unsupported design option: {design}. Choose from: {', '.join(_DESIGNS)}")

    coded = np.asarray(coded, dtype=float)
    # Bounds are the extreme settings. Designs with points beyond +/-1 (e.g. a
    # circumscribed CCD with face='ccc') are shrunk so the extreme points lie on the
    # bounds, which makes them inscribed designs.
    extent = np.abs(coded).max(axis=0) if len(coded) else np.ones(k)
    extent[extent <= 1] = 1.0
    return coded / extent


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
        Bounds for each input factor. Each row is [xmin, xmax]. Designs are
        generated in coded units where -1 maps to xmin and +1 maps to xmax, and no
        generated experiment lies outside the bounds. If None, the design is
        returned in coded units.
    design : str, default='bbdesign'
        Type of experimental design. Options are:

        - 'fullfact': Full factorial design (requires ``levels=[...]``); the
          lowest/highest level of each factor map to the bounds.
        - 'ff2n': 2-level full factorial design
        - 'pbdesign': Plackett-Burman design
        - 'gsd': Generalized subset design (requires ``levels`` and ``reduction``)
        - 'bbdesign': Box-Behnken design. It needs at least 3 inputs; with 1 or 2
          inputs a face-centered central composite design is used instead and a
          UserWarning is issued.
        - 'ccdesign': Central composite design. Face-centered (``face='ccf'``,
          alpha = 1, three levels per factor) by default so that every run is inside
          the bounds. If you request ``face='ccc'`` the design is shrunk so its axial
          points lie on the bounds (i.e. it becomes an inscribed design).
        - 'lhs': Latin hypercube sampling (use ``samples=n``)
    model : sklearn estimator, optional
        Custom model to use. If None (default), uses a pipeline with
        a :class:`CodedScaler`, 2nd-order PolynomialFeatures, and LinearRegressionUQ.
    design_kwargs : dict, optional
        Keyword arguments passed to the pyDOE3 design function.
    **kwargs
        Additional keyword arguments passed to the pyDOE3 design function. They
        are merged into ``design_kwargs``.

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

    Notes
    -----
    With the default model, the fitted coefficients are in *coded* units: each input
    is scaled so that its bounds map to [-1, 1] before the polynomial features are
    built. A coefficient therefore is the change in the response per half-range of
    that input. If ``bounds`` is None, the minimum and maximum of the fitted data
    define the coded units instead.

    Examples
    --------
    >>> sr = SurfaceResponse(
    ...     inputs=['temperature', 'pressure', 'time'],
    ...     outputs=['yield'],
    ...     bounds=[[100, 200], [1, 5], [10, 60]],
    ...     design='bbdesign'
    ... )
    >>> design_df = sr.generate_design()
    >>> # Run experiments and collect data
    >>> sr.set_results([[0.75], [0.82], [0.68], ...])
    >>> sr.fit()
    >>> print(sr.summary())
    """

    design = _DesignParam()

    def __init__(
        self,
        inputs=None,
        outputs=None,
        bounds=None,
        design="bbdesign",
        model=None,
        design_kwargs=None,
        **kwargs,
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

        # Validate and set bounds. np.asarray does not copy an ndarray, so a cloned
        # estimator stores exactly the object it was given (sklearn convention).
        if bounds is not None:
            bounds = np.asarray(bounds)
            if bounds.shape != (len(inputs), 2):
                raise ValueError(f"bounds must have shape ({len(inputs)}, 2), got {bounds.shape}")
            if np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ValueError("All bounds must have min < max")
        self.bounds = bounds

        self.design = design
        self.design_kwargs = {**(design_kwargs or {}), **kwargs} if kwargs else design_kwargs

        # Generate the design (in coded units) now so bad options fail early.
        self._design = self._make_design()

        self.model = model

        # Initialize pipeline
        if model is None:
            self.default = True
            scaler = CodedScaler(bounds=bounds).set_output(transform="pandas")
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

    def _design_key(self):
        return (str(self.design), repr(self.design_kwargs), len(self.inputs))

    def _make_design(self):
        coded = _coded_design(str(self.design), len(self.inputs), self.design_kwargs or {})
        self._design_key_ = self._design_key()
        return coded

    def __getitem__(self, ind):
        """Index the pipeline; slices return a plain sklearn Pipeline."""
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return Pipeline(self.steps[ind], memory=self.memory, verbose=self.verbose)
        return super().__getitem__(ind)

    def generate_design(self, shuffle=True):
        """Create a design dataframe with experimental conditions.

        Generates the experimental design matrix by mapping the coded design
        points ([-1, 1]) to the actual factor bounds.

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
        ...     bounds=[[100, 200], [1, 5]],
        ...     design='ccdesign',
        ... )
        >>> design = sr.generate_design(shuffle=True)
        """
        if getattr(self, "_design_key_", None) != self._design_key():
            # Parameters were changed with set_params
            self._design = self._make_design()

        design = self._design.copy()

        if self.bounds is not None:
            b = np.asarray(self.bounds, dtype=float)
            mins, maxs = b[:, 0], b[:, 1]
            # coded -1 -> xmin, +1 -> xmax
            design = (design + 1) * (maxs - mins) / 2 + mins

        df = pd.DataFrame(data=design, columns=self.inputs)

        if shuffle:
            df = df.sample(frac=1)

        self.input = df
        return df

    def set_results(self, data):
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
            If generate_design() has not been called yet.

        Examples
        --------
        >>> sr.generate_design()
        >>> results = [[0.75], [0.82], [0.68]]  # From experiments
        >>> sr.set_results(results)
        """
        if not hasattr(self, "input"):
            raise AttributeError("Must call generate_design() before set_results()")

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

    def set_output(self, data=None, *, transform=None):
        """Configure sklearn output containers, or (deprecated) set response data.

        ``sr.set_output(transform="pandas")`` behaves exactly like
        :meth:`sklearn.pipeline.Pipeline.set_output`.

        Calling ``sr.set_output(data)`` with experimental results is deprecated;
        use :meth:`set_results` instead.
        """
        if data is None:
            return super().set_output(transform=transform)
        if transform is not None:
            raise TypeError("Pass either response data or transform=..., not both")
        warnings.warn(
            "SurfaceResponse.set_output(data) is deprecated; use set_results(data). "
            "set_output(transform=...) now has its standard sklearn meaning.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.set_results(data)

    def fit(self, X=None, y=None):
        """Fit the response surface model to the experimental data.

        Parameters
        ----------
        X : array-like, optional
            Input data. If None, uses self.input from generate_design().
        y : array-like, optional
            Output data. If None, uses self.output from set_results().

        Returns
        -------
        self
            Fitted estimator.
        """
        if X is None and y is None:
            if not hasattr(self, "input") or not hasattr(self, "output"):
                raise AttributeError("Must set input and output data before fitting")
            X, y = self.input, self.output
        if self.default:
            # keep the coded-unit scaling in sync with the declared bounds
            self.steps[0][1].bounds = self.bounds
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
        considered statistically significant ("yes"), otherwise "no".

        Coefficients are reported in coded units: each input is scaled so that
        its bounds map to [-1, 1] (or its observed min/max if no bounds were
        given) before the polynomial features are computed.
        """
        X, y = self.input, self.output

        s = [f"{len(X)} data points"]
        yp = self.predict(X)
        # Ensure yp has the same shape as y for proper subtraction
        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)
        # Convert yp to DataFrame to match y's structure for proper pandas operations
        if hasattr(y, "columns"):
            yp = pd.DataFrame(yp, columns=y.columns, index=y.index)
        errs = y - yp

        if self.default:
            features = self["poly"].get_feature_names_out()

            pars = self["surface response"].coefs_
            pars_cint = self["surface response"].pars_cint_
            pars_se = self["surface response"].pars_se_

            nrows, ncols = pars.shape

            mae = [float(self._sigfig(x)) for x in (np.abs(errs).mean())]
            rmse = [float(self._sigfig(x)) for x in np.sqrt((errs**2).mean())]

            s += [f"  R² score: {self.score(X, y):.4f}"]
            s += [
                f"  MAE  = {mae}",
                "",
                f"  RMSE = {rmse}",
                "",
            ]
            if self.bounds is not None:
                s += ["Coefficients are in coded units: each input's bounds map to [-1, 1].", ""]
            else:
                s += [
                    "Coefficients are in coded units: each input's observed min/max map to "
                    "[-1, 1].",
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
                            "yes" if pars_cint[i][j][0] * pars_cint[i][j][1] > 0 else "no",
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

            mae = [float(self._sigfig(x)) for x in (np.abs(errs).mean())]
            rmse = [float(self._sigfig(x)) for x in np.sqrt((errs**2).mean())]

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
