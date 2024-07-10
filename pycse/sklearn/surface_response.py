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
    """A class for a Surface Response design of experiment."""

    def __init__(
        self, inputs=None, outputs=None, bounds=None, design="bbdesign", model=None, **kwargs
    ):
        """inputs : list of strings, name of each factor

        outputs : list of strings, name of each response

        bounds : 2D array, Each row is [xmin, xmax] for a component.
        This assumes that [-1, 1] map to the bounds.

        design: string, one of the designs in pyDOE3
        fullfact, ff2n, pbdesign, gsd, bbdesign, ccdesign, lhs
        except fracfact. That one is confusing and I don't know how you use it.

        model : an sklearn model, defaults to scaled, order-2 polynomial
        features with linear regression.

        kwargs are passed to the pyDOE3 model

        Builds a linear regression model. The polynomial features are
        automatically generated up to the specified order.

        """
        self.inputs = inputs
        self.outputs = outputs
        self.bounds = np.array(bounds)
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
            self._design == ccdesign(len(inputs), **kwargs)
        elif design == "lhs":
            self._design = lhs(len(inputs), **kwargs)
        else:
            raise Exception(f"Unsupported design option: {design}")

        self.model = model

        if model is None:
            self.default = True
            super().__init__(
                steps=[
                    ("minmax", MinMaxScaler(feature_range=(-1, 1))),
                    ("poly", PolynomialFeatures(2)),
                    ("surface response", LinearRegressionUQ()),
                ]
            )
        else:
            self.default = False
            super().__init__(steps=[("usermodel", model)])

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
        """Set output to data.

        data : list or array of results. Each row should correspond to the same
        row in the input.

        """
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

    def parity(self):
        """Creates parity plot between true values and predicted values."""
        X, y = self.input, self.output
        pred = self.predict(X)
        plt.scatter(y, pred)
        plt.plot(np.linspace(y.min(), y.max()), np.linspace(y.min(), y.max()))

        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")

        return plt.gcf()

    def _sigfig(self, x, n=3):
        """Round X to N significant figures."""
        # Adapted from https://gist.github.com/ttamg/3f65227fd580b3d8dc8ba91e01507280
        return np.round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))

    def summary(self):
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
            rmse = [self._sigfig(x) for x in (errs**2).mean()]

            s += [f"  score: {self.score(X, y)}"]
            s += [
                f"  mae  = {(mae)}",
                "",
                f"  rmse = {rmse}",
                "",
            ]

            for i in range(ncols):
                data = []
                s += [f"Output_{i} = {y.columns[i]}"]
                for j, name in enumerate(features):
                    data += [
                        [
                            f"{name}_{i}",
                            pars[j][i],
                            pars_cint[0][j][i],
                            pars_cint[1][j][i],
                            pars_se[j][i],
                            np.sign(pars_cint[0][j][i] * pars_cint[1][j][i]) > 0,
                        ]
                    ]
                s += [
                    tabulate.tabulate(
                        data,
                        headers=["var", "value", "ci_lower", "ci_upper", "se", "significant"],
                        tablefmt="orgtbl",
                    )
                ]
                s += [""]
        else:
            s += ["User defined model:", repr(self["usermodel"])]

            mae = [self._sigfig(x) for x in (np.abs(errs).mean())]
            rmse = [self._sigfig(x) for x in (errs**2).mean()]

            s += [f"  score: {self.score(X, y)}"]
            s += [
                f"  mae  = {(mae)}",
                "",
                f"  rmse = {rmse}",
                "",
            ]

        return "\n".join(s)
