"""Python calculations in Science and Engineering.

Pycse is compatible with Python 3.6.

"""

__version__ = '2.0.0'

# * Setup inline images for IPython
# Make inline figures the default
try:
    from IPython import get_ipython
    from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic)

    from IPython.core.pylabtools import backends
    import matplotlib as mpl
    mpl.interactive(True)
    mpl.use(backends['inline'])
except:
    pass

# * IPython magic for pycse
# IPython magic functions.
# Adapated from
# http://ipython.readthedocs.io/en/stable/config/custommagics.html

try:
    @register_line_magic
    def pycse_test(line):
        PASSED = True
        import pycse
        s = []
        s += ['pycse version: {0}'.format(pycse.__version__)]

        import numpy
        s += ['numpy version: {0}'.format(numpy.__version__)]

        import scipy
        s += ['scipy version: {0}'.format(scipy.__version__)]

        import matplotlib
        s += ['matplotlib version: {0}'.format(matplotlib.__version__)]

        import IPython
        s += ['IPython version: {0}'.format(IPython.__version__)]

        return '\n'.join(s)
except:
    pass


# * load some common libraries

# The goal here is to make it easy for beginners to get started with a minimal
# amount of importing libraries. This is not what most expert programmers would
# do, and it is not for them. This tries to mimic the behavior of Matlab where
# every function you want is in the global namespace. This is a transition
# behavior to allow students to focus on problem solving and not on Python
# library imports.

import numpy as np
from numpy import array, diag, dot, mean, polyfit, polyval, polyder, polyint, std, transpose
import numpy.linalg as la
from numpy.linalg import det, eig, eigvals, inv, solve, svd

import scipy as sp
import matplotlib.pyplot as plt

from scipy.integrate import quad, solve_ivp, solve_bvp
from scipy.interpolate import interp1d

from .PYCSE import polyfit, regress, nlinfit, Rsquared
from .utils import feq, flt, fgt, fle, fge

from .beginner import *

try:
    from .colab import *
except:
    pass
