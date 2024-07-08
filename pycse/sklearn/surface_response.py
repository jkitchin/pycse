from sklearn.preprocessing import PolynomialFeatures
from pycse.sklearn.lr_uq import LinearRegressionUQ
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from pyDOE3 import bbdesign
import numpy as np
import pandas as pd
import tabulate


class SurfaceResponse(Pipeline):
    """A class for a Surface Response design of experiment.

    TODO: can we make it more flexible on the design, e.g. to select all the
    options from pyDOE3? I need to check the signatures of these to see how
    compatible they are.

    """

    def __init__(self, inputs=None, outputs=None, bounds=None, order=2, **kwargs):

        """inputs : list of strings, name of each factor

        outputs : list of strings, name of each response

        bounds : 2D array, Each row is [xmin, xmax] for a component.

        order: int, polynomial model order

        kwargs are passed to pyDOE3.bbenken

        Builds a linear regression model. The polynomial features are
        automatically generated.

        """
        self.inputs = inputs
        self.outputs = outputs
        self.bounds = np.array(bounds)
        self._design = bbdesign(len(inputs), **kwargs)
        self.order = order
        super().__init__(
            steps=[
                ("poly", PolynomialFeatures(order)),
                ("surface response", LinearRegressionUQ()),
            ]
        )

    def design(self, shuffle=True):
        """Creates a design dataframe.

        shuffle: Boolean, if true shuffle the results.

        Returns
        -------
        a data frame

        """
        design = self._design
        nrows, ncols = design.shape

        # with lrange we assume that (-1, 1) maps to (xmin, xmax)
        # here a=-1, b=1
        # Xsc = a + (x - xmin) * (b - a) / (xmax - xmin)

        # solving for x
        # x = (Xsc - a) * (xmax - xmin) / (b - a) + xmin

        a, b = -1, 1

        # lrange is column wise min, max
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
        index = self.input.index
        df = pd.DataFrame(data, index=index, columns=self.outputs)
        self.output = df
        return self.output

    def fit(self, X=None, y=None):
        X, y = self.input, self.output
        return super().fit(X, y)

    def score(self, X=None, y=None):
        X, y = self.input, self.output
        return super().score(X, y)

    # No need to define fit/predict here, we get them from Pipeline

    def parity(self):
        """Creates parity plot between true values and predicted values."""
        X, y = self.input, self.output
        pred = self.predict(X)
        plt.scatter(y, pred)
        plt.plot(np.linspace(y.min(), y.max()), np.linspace(y.min(), y.max()))

        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")

        return plt.gcf()

    def summary(self):
        X, y = self.input, self.output

        s = [f"{len(X)} data points"]
        yp = self.predict(X)
        errs = y - yp

        features = self["poly"].get_feature_names_out()

        pars = self["surface response"].coefs_
        pars_cint = self["surface response"].pars_cint
        pars_se = self["surface response"].pars_se

        nrows, ncols = pars.shape

        s += [f"  score: {self.score(X, y)}"]
        s += [
            f"  mae = {(np.abs(errs).mean().tolist())}",
            "",
            f"  rmse = {(errs**2).mean().tolist()}",
            "",
        ]

        for i in range(ncols):
            data = []
            s += [f"Output_{i} = {y.columns[i]}"]
            for j, name in enumerate(features):
                data += [
                    [
                        f"{name}_{i}",
                        pars_cint[0][j][i],
                        pars_cint[1][j][i],
                        pars_se[j][i],
                        np.sign(pars_cint[0][j][i] * pars_cint[1][j][i]) > 0,
                    ]
                ]
            s += [
                tabulate.tabulate(
                    data,
                    headers=["var", "ci_lower", "ci_upper", "se", "significant"],
                    tablefmt="orgtbl",
                )
            ]
            s += [""]

        return "\n".join(s)
