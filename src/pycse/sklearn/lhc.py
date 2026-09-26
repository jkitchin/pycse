"""Latin Square Design of Experiments.

Despite the module name (``lhc``), this module implements **Latin square**
designs, not Latin hypercube sampling. For Latin hypercube sampling use
``SurfaceResponse(..., design="lhs")`` or ``pyDOE3.lhs``.

This module provides classes for creating and analyzing Latin Square
experimental designs. A Latin Square is a square array filled with different
symbols such that each symbol occurs exactly once in each row and column.

This is particularly useful for experiments with three factors where you want to
efficiently explore the factor space while controlling for row and column effects.

Classes
-------
LatinSquare
    Create, analyze, and predict from Latin Square experimental designs.

Notes
-----
This implementation is designed for three-factor systems with equal numbers of
levels (typically 3 or 4 levels per factor). The design uses:
- First factor: rows
- Second factor: columns
- Third factor: cell values (Latin Square property)

Examples
--------
>>> vars_dict = {
...     'Temperature': [100, 150, 200],
...     'Pressure': [1, 2, 3],
...     'Catalyst': ['A', 'B', 'C']
... }
>>> ls = LatinSquare(vars=vars_dict)
>>> design = ls.design()
>>> # Run experiments and collect data
>>> y = pd.Series([...], name='Yield')
>>> ls.fit(design, y)
>>> ls.anova()
"""

import numpy as np
import pandas as pd
import scipy.stats as stats


class LatinSquare:
    """Latin Square experimental design with ANOVA analysis.

    A Latin Square design is used for experiments with three factors where
    each factor has the same number of levels. The design ensures that each
    level of each factor appears exactly once with each level of the other factors.

    This is a Latin *square* (three categorical factors, p levels each,
    p**2 runs), not Latin hypercube sampling.

    Parameters
    ----------
    vars : dict
        Dictionary mapping factor names to lists of factor levels.
        Must contain exactly 3 factors, each with the same number of levels.
        - First entry: row factor
        - Second entry: column factor
        - Third entry: cell factor (Latin Square values)
    random_state : int, numpy.random.Generator, or None, default=None
        Seed or generator used when ``design(shuffle=True)`` randomizes the
        design. If None, the class attribute ``seed`` (42) is used, so results are
        reproducible by default. The global NumPy random state is never modified.

    Attributes
    ----------
    vars : dict
        The factor dictionary provided during initialization
    labels : list
        Factor names extracted from vars keys
    effects : dict, optional
        Factor effects computed by fit(). Available after calling fit().
    results : DataFrame, optional
        Detailed results including effects and residuals. Available after fit().
    y : str, optional
        Name of the response variable. Available after fit().

    Examples
    --------
    Create a Latin Square design for a chemical process:

    >>> vars_dict = {
    ...     'Temperature': [100, 150, 200],
    ...     'Pressure': [1, 2, 3],
    ...     'Catalyst': ['A', 'B', 'C']
    ... }
    >>> ls = LatinSquare(vars=vars_dict)
    >>> design = ls.design()
    >>> print(design)
       Temperature  Pressure Catalyst
    0          100         1        A
    1          100         2        B
    2          100         3        C
    ...

    After running experiments:

    >>> import pandas as pd
    >>> y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name='Yield')
    >>> results = ls.fit(design, y)
    >>> anova_table = ls.anova()
    >>> prediction = ls.predict([150, 2, 'B'])

    Notes
    -----
    Shuffling uses a private ``numpy.random.Generator`` created from
    ``random_state`` (default: the class-level ``seed`` attribute, 42), so
    ``design(shuffle=True)`` is reproducible for a given seed and does not touch
    the global NumPy random state.
    """

    seed = 42

    def __init__(self, vars=None, random_state=None):
        """Initialize the LatinSquare class.

        Parameters
        ----------
        vars : dict
            Dictionary mapping factor names to lists of factor levels.
            Must contain exactly 3 factors with equal numbers of levels.
        random_state : int, numpy.random.Generator, or None, default=None
            Seed for shuffling. None uses the class attribute ``seed``.

        Raises
        ------
        ValueError
            If vars is None, not a dict, doesn't have exactly 3 factors,
            or factors don't all have the same number of levels.
        TypeError
            If vars is not a dictionary.

        Examples
        --------
        >>> vars_dict = {
        ...     'Temperature': [100, 150, 200],
        ...     'Pressure': [1, 2, 3],
        ...     'Catalyst': ['A', 'B', 'C']
        ... }
        >>> ls = LatinSquare(vars=vars_dict)
        """
        # Validate input
        if vars is None:
            raise ValueError("vars parameter is required and cannot be None")

        if not isinstance(vars, dict):
            raise TypeError(f"vars must be a dictionary, got {type(vars).__name__}")

        if len(vars) != 3:
            raise ValueError(
                f"Latin Square requires exactly 3 factors, got {len(vars)}. "
                "Factors provided: " + ", ".join(vars.keys())
            )

        # Check all factors have the same number of levels
        level_counts = [len(levels) for levels in vars.values()]
        if len(set(level_counts)) > 1:
            factor_info = [f"{k}: {len(v)} levels" for k, v in vars.items()]
            raise ValueError(
                f"All factors must have the same number of levels. Got: {', '.join(factor_info)}"
            )

        # Check minimum level count
        if level_counts[0] < 2:
            raise ValueError(f"Each factor must have at least 2 levels, got {level_counts[0]}")

        self.vars = vars
        self.labels = list(vars.keys())
        self.random_state = random_state
        self._rng = np.random.default_rng(self.seed if random_state is None else random_state)

    def design(self, shuffle=False):
        """Generate the Latin Square experimental design matrix.

        Creates a design matrix where each row represents one experimental run.
        The design ensures that each level of the third factor (cell factor)
        appears exactly once in each row and column.

        Parameters
        ----------
        shuffle : bool, default=False
            If True, randomize the order of experimental runs.

        Returns
        -------
        DataFrame
            Design matrix with one row per experiment. Columns correspond to
            the three factors in the order they were defined in vars.

        Examples
        --------
        >>> vars_dict = {'A': [1, 2, 3], 'B': [10, 20, 30], 'C': ['X', 'Y', 'Z']}
        >>> ls = LatinSquare(vars=vars_dict)
        >>> design = ls.design(shuffle=False)
        >>> print(design)
           A   B  C
        0  1  10  X
        1  1  20  Y
        2  1  30  Z
        ...

        >>> shuffled = ls.design(shuffle=True)  # Randomized run order
        """
        # Setup the Latin Square using cyclic permutation
        lhs = []
        levels = list(self.vars[self.labels[2]])

        if shuffle:
            levels = [levels[i] for i in self._rng.permutation(len(levels))]

        lhs.append(levels)

        # Create cyclic permutations for remaining rows
        for i in range(1, len(levels)):
            levels = levels[1:] + levels[:1]  # Rotate left
            lhs.append(levels)

        # Build experiment list
        expts = []
        for i, row_level in enumerate(self.vars[self.labels[0]]):
            for j, col_level in enumerate(self.vars[self.labels[1]]):
                expts.append([row_level, col_level, lhs[i][j]])

        df = pd.DataFrame(expts, columns=self.labels)

        if shuffle:
            return df.iloc[self._rng.permutation(len(df))].reset_index(drop=True)
        else:
            return df

    def pivot(self):
        """Create a pivot table view of the Latin Square design.

        Returns
        -------
        DataFrame
            Pivot table with first factor as rows, second factor as columns,
            and third factor as cell values, showing the Latin Square structure.

        Examples
        --------
        >>> vars_dict = {'Row': [1, 2, 3], 'Col': ['A', 'B', 'C'], 'Treatment': ['X', 'Y', 'Z']}
        >>> ls = LatinSquare(vars=vars_dict)
        >>> print(ls.pivot())
        Col      A  B  C
        Row
        1        X  Y  Z
        2        Y  Z  X
        3        Z  X  Y
        """
        df = self.design()[self.labels]
        return df.pivot(index=self.labels[0], columns=self.labels[1])

    def fit(self, X, y):
        """Fit the Latin Square model to experimental data.

        Performs an additive effects analysis, computing the main effect of
        each factor and residuals.

        Parameters
        ----------
        X : DataFrame
            Experimental design matrix (from design() method).
            Must contain columns matching the factor names.
        y : Series
            Response values for each experiment. Must have a name attribute.

        Returns
        -------
        DataFrame
            Results dataframe containing original data plus computed effects
            and residuals for each experiment.

        Raises
        ------
        ValueError
            If X doesn't contain the required factor columns or if lengths don't match.
        AttributeError
            If y doesn't have a name attribute.

        Examples
        --------
        >>> ls = LatinSquare(vars={'A': [1, 2, 3], 'B': [10, 20, 30], 'C': ['X', 'Y', 'Z']})
        >>> design = ls.design()
        >>> y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name='Response')
        >>> results = ls.fit(design, y)
        >>> print(ls.effects['avg'])  # Overall average
        >>> print(ls.effects['A'])    # Row effects
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y).__name__}")

        if not hasattr(y, "name") or y.name is None:
            raise AttributeError("y must have a name attribute (use pd.Series(..., name='...'))")

        # Check all required columns present
        missing_cols = set(self.labels) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"X is missing required columns: {missing_cols}. Expected columns: {self.labels}"
            )

        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")

        # Combine design and response
        df = pd.concat([X, y], axis=1)

        # Compute overall average
        avg = df[y.name].mean()
        df["avg"] = avg

        # Compute main effects for each factor
        rows = df.groupby(self.labels[0])[y.name].mean() - avg
        cols = df.groupby(self.labels[1])[y.name].mean() - avg
        effs = df.groupby(self.labels[2])[y.name].mean() - avg

        # Store effects (as Series for later use)
        self.effects = {
            "avg": avg,
            self.labels[0]: rows,
            self.labels[1]: cols,
            self.labels[2]: effs,
        }

        # Map effects back to dataframe
        df[f"{self.labels[0]}_effect"] = df[self.labels[0]].map(rows)
        df[f"{self.labels[1]}_effect"] = df[self.labels[1]].map(cols)
        df[f"{self.labels[2]}_effect"] = df[self.labels[2]].map(effs)

        # Compute residuals
        df["residuals"] = (
            df[y.name]
            - df["avg"]
            - df[f"{self.labels[0]}_effect"]
            - df[f"{self.labels[1]}_effect"]
            - df[f"{self.labels[2]}_effect"]
        )

        self.results = df
        self.y = y.name
        return self.results

    def anova(self, alpha=0.05):
        """Perform ANOVA to test significance of factor effects.

        Uses the standard Latin square ANOVA: with p levels and N = p**2 runs,
        each factor has p - 1 degrees of freedom and the residual has
        (p - 1)(p - 2). F = MS_factor / MS_residual and the p-value is
        ``scipy.stats.f.sf(F, p - 1, (p - 1)(p - 2))``.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level used for the critical F value and the
            ``Significant`` column.

        Returns
        -------
        DataFrame
            ANOVA table with one row per factor effect plus the residuals, and
            numeric columns:
            - ``"<y> effect"``: effect name (str)
            - ``"df"``: degrees of freedom (int)
            - ``"sum_sq"``: sum of squares
            - ``"mean_sq"``: mean square
            - ``"F-score (fc=...)"``: F statistic (NaN for residuals)
            - ``"p-value"``: P(F > F-score) (NaN for residuals)
            - ``"Significant"``: bool, p-value < alpha

        Raises
        ------
        AttributeError
            If fit() has not been called yet.

        Examples
        --------
        >>> # After fitting the model
        >>> anova_results = ls.anova()

        Notes
        -----
        With only 2 levels there are no residual degrees of freedom, so F and
        p-values are NaN.
        """
        if not hasattr(self, "results"):
            raise AttributeError(
                "Must call fit() before anova(). The model needs to be fitted to data first."
            )

        df = self.results
        p = len(self.vars[self.labels[0]])
        df_effect = p - 1
        df_resid = len(df) - 1 - 3 * df_effect  # (p - 1)(p - 2) for a full square

        effect_cols = [f"{label}_effect" for label in self.labels] + ["residuals"]
        dofs = np.array([df_effect] * 3 + [df_resid])

        # Each observation carries its level's effect, so summing effect**2 over all
        # runs gives the usual sum of squares p * sum(effect_level**2).
        sum_sq = (df[effect_cols].astype(float) ** 2).sum().to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_sq = np.where(dofs > 0, sum_sq / np.where(dofs > 0, dofs, 1), np.nan)
            f_scores = mean_sq / mean_sq[-1]
        f_scores[-1] = np.nan

        if df_resid > 0:
            p_values = stats.f.sf(f_scores, df_effect, df_resid)
            fc = stats.f.ppf(1 - alpha, df_effect, df_resid)
        else:
            p_values = np.full(4, np.nan)
            fc = np.nan

        return pd.DataFrame(
            {
                f"{self.y} effect": effect_cols,
                "df": dofs.astype(int),
                "sum_sq": sum_sq,
                "mean_sq": mean_sq,
                f"F-score (fc={fc:1.1f})": f_scores,
                "p-value": p_values,
                "Significant": np.nan_to_num(p_values, nan=1.0) < alpha,
            }
        )

    def predict(self, args):
        """Predict response for a given combination of factor levels.

        Uses the additive effects model to predict the response:
        y = average + effect_1 + effect_2 + effect_3

        Parameters
        ----------
        args : list or array-like
            Factor levels in order [row_level, col_level, cell_level].
            Must be levels that exist in the original design.

        Returns
        -------
        float
            Predicted response value.

        Raises
        ------
        AttributeError
            If fit() has not been called yet.
        ValueError
            If args doesn't have exactly 3 elements or contains unknown levels.

        Examples
        --------
        >>> # After fitting the model
        >>> prediction = ls.predict([150, 2, 'B'])
        >>> print(f"Predicted response: {prediction:.2f}")

        Notes
        -----
        This method only works for interpolation (factor levels seen during
        fitting). It does not extrapolate to new levels.
        """
        if not hasattr(self, "effects"):
            raise AttributeError(
                "Must call fit() before predict(). The model needs to be fitted to data first."
            )

        if len(args) != 3:
            raise ValueError(
                f"predict() requires exactly 3 arguments (one per factor), got {len(args)}"
            )

        # Start with overall average
        prediction = self.effects["avg"]

        # Add each factor effect
        for i, val in enumerate(args):
            label = self.labels[i]

            # Check if this level exists
            if val not in self.results[label].values:
                raise ValueError(
                    f"Unknown level '{val}' for factor '{label}'. "
                    f"Valid levels: {sorted(self.results[label].unique())}"
                )

            # Get the effect for this level
            mask = self.results[label] == val
            effect = self.results[mask][f"{label}_effect"].iloc[0]
            prediction += effect

        return prediction

    def __repr__(self):
        """Return detailed string representation."""
        n_factors = len(self.labels)
        n_levels = len(self.vars[self.labels[0]])
        n_expts = n_levels**2

        fitted_str = ""
        if hasattr(self, "results"):
            fitted_str = f", fitted with {len(self.results)} observations"

        return (
            f"LatinSquare(factors={n_factors}, levels={n_levels}, "
            f"experiments={n_expts}{fitted_str})"
        )

    def __str__(self):
        """Return readable string description."""
        n_levels = len(self.vars[self.labels[0]])
        factor_str = ", ".join([f"'{label}'" for label in self.labels])

        status = "not fitted"
        if hasattr(self, "results"):
            status = f"fitted ({len(self.results)} obs)"

        return (
            f"Latin Square Design:\n"
            f"  Factors: {factor_str}\n"
            f"  Levels per factor: {n_levels}\n"
            f"  Total experiments: {n_levels**2}\n"
            f"  Status: {status}"
        )
