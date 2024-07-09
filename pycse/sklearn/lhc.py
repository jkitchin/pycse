"""
An sklearn-compatible Latin Hypercube library.

This is lightly tested on three-factor systems with 3 and 4 levels.

"""

import numpy as np
import pandas as pd
import scipy.stats as stats


class LatinSquare:
    seed = 42

    def __init__(self, vars=None):
        """vars is a dictionary: key: values
        The first entry is used for rows
        The second entry is used for cols
        The third entry is used in the cells.

        The values are the levels for each factor

        Each entry should have the same number of elements. Tested on 3 and 4
        levels for three factors.

        """

        self.vars = vars
        self.labels = list(vars.keys())
        np.random.seed(self.seed)

    def design(self, shuffle=False):
        """Returns a design matrix as a dataframe.

        Each row is an experiment to run. If shuffle is True, it is randomized.

        """
        # Setup the design matrix
        lhs = []
        levels = self.vars[self.labels[2]].copy()
        if shuffle:
            np.random.shuffle(levels)
        lhs += [levels]
        for i in range(1, len(levels)):
            levels = levels[1:] + levels[:1]
            lhs += [levels]

        expts = []

        for i, r in enumerate(self.vars[self.labels[0]]):
            for j, c in enumerate(self.vars[self.labels[1]]):
                expts += [[r, c, lhs[i][j]]]

        df = pd.DataFrame(expts, columns=self.labels)

        if shuffle:
            return df.sample(frac=1)
        else:
            return df

    def pivot(self):
        """Show the Pivot table version of the design."""

        df = self.design()[self.labels]
        return df.pivot(index=self.labels[0], columns=self.labels[1])

    def fit(self, X, y):
        """X is the experiment design dataframe

        y is a pd.Series for the responses.

        This signature is for compatibility with the sklearn fit API.

        """
        df = pd.concat([X, y], axis=1)

        avg = df["avg"] = df[y.name].mean()
        rows = df.groupby(self.labels[0])[y.name].mean() - avg
        cols = df.groupby(self.labels[1])[y.name].mean() - avg
        effs = df.groupby(self.labels[2])[y.name].mean() - avg
        resids = df[y.name] - avg - rows - cols - effs

        # the values are series here.
        self.effects = {
            "avg": avg,
            self.labels[0]: rows,
            self.labels[1]: cols,
            self.labels[2]: effs,
            "residuals": resids,
        }

        df[f"{self.labels[0]}_effect"] = df.join(
            rows, on=self.labels[0], rsuffix="_effect", how="left"
        )[f"{y.name}_effect"]
        df[f"{self.labels[1]}_effect"] = df.join(
            cols, on=self.labels[1], rsuffix="_effect", how="left"
        )[f"{y.name}_effect"]
        df[f"{self.labels[2]}_effect"] = df.join(
            effs, on=self.labels[2], rsuffix="_effect", how="left"
        )[f"{y.name}_effect"]
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

    def anova(self):
        """Do an ANOVA on significance of each effect.

        Returns a dataframe containing normalized variance for each effect, and
        significance.

        """
        df = self.results
        N = len(df) ** 2
        n0 = len(self.vars[self.labels[0]]) - 1
        n1 = len(self.vars[self.labels[1]]) - 1
        n2 = len(self.vars[self.labels[2]]) - 1

        dof = np.array([n0, n1, n2, len(df) - 1 - n0 - n1 - n2])
        ddof = N / (N - (N - dof))

        # These are variances in each factor
        S2 = (
            df[
                [
                    f"{self.labels[0]}_effect",
                    f"{self.labels[1]}_effect",
                    f"{self.labels[2]}_effect",
                    "residuals",
                ]
            ]
        ).var(ddof=0) * ddof

        # I think this is like an F-score
        ns = S2 / S2.loc["residuals"]

        # Critical F-score. Bigger than this means significant
        fc = stats.f.ppf(0.95, dof[0], dof[-1])

        table = np.vstack([ns.index.values, ns.values, ns > fc]).T

        return pd.DataFrame(
            table, columns=[f"{self.y} effect", f"F-score (fc={fc:1.1f})", "Significant"]
        )

    def predict(self, args):
        """Predict the response for ARGS. ARGS is a list of labels in order
        (row, col, effect). This only works for levels you have already defined,
        i.e. it is not a continuous function.

        """
        p = self.effects["avg"]
        for i, val in enumerate(args):
            label = self.labels[i]
            ind = self.results[label] == val
            effect = self.results[ind][f"{label}_effect"].mean()
            p += effect
        return p
