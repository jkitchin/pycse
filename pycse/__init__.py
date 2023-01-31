"""Python calculations in Science and Engineering.

Pycse is compatible with Python 3.6+.

"""

__version__ = "2.1.9"

# * Setup inline images for IPython
# Make inline figures the default

from .PYCSE import (
    polyfit,
    regress,
    predict,
    nlinfit,
    nlpredict,
    Rsquared,
    bic,
    lbic,
)
from .utils import feq, flt, fgt, fle, fge, read_gsheet

from .hashcache import hashcache

# from .beginner import *


from IPython import get_ipython
from IPython.core.magic import (
    register_line_magic,
    register_cell_magic,
    register_line_cell_magic,
)

# * load some common libraries
# The goal here is to make it easy for beginners to get started with a minimal
# amount of importing libraries. This is not what most expert programmers would
# do, and it is not for them. This tries to mimic the behavior of Matlab where
# every function you want is in the global namespace. This is a transition
# behavior to allow students to focus on problem solving and not on Python
# library imports.

import pycse
import numpy
import matplotlib
import IPython
import scipy

# import numpy as np
# from numpy import (
#     array,
#     diag,
#     dot,
#     mean,
#     polyfit,
#     polyval,
#     polyder,
#     polyint,
#     std,
#     transpose,
# )
# import numpy.linalg as la
# from numpy.linalg import det, eig, eigvals, inv, solve, svd

# import scipy as sp
# import matplotlib.pyplot as plt

# from scipy.integrate import quad, solve_ivp, solve_bvp
# from scipy.optimize import fsolve, root
# from scipy.interpolate import interp1d

# from IPython.core.pylabtools import backends
# import matplotlib as mpl

# mpl.interactive(True)
# mpl.use(backends["inline"])


# * IPython magic for pycse
# IPython magic functions.
# Adapated from
# http://ipython.readthedocs.io/en/stable/config/custommagics.html

try:

    @register_line_magic
    def pycse_test(line):
        """Print the versions of libraries installed."""

        s = []
        s += ["pycse version: {0}".format(pycse.__version__)]

        s += ["numpy version: {0}".format(numpy.__version__)]

        s += ["scipy version: {0}".format(scipy.__version__)]

        s += ["matplotlib version: {0}".format(matplotlib.__version__)]

        s += ["IPython version: {0}".format(IPython.__version__)]

        return "\n".join(s)

except:  # noqa: E722
    pass


# We try this because it fails if you are not in colab, e.g. in a regular
# Jupyter notebook. ModuleNotFoundError: No module named 'google.colab'
try:
    from .colab import *
except ModuleNotFoundError:
    pass
