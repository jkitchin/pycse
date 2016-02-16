from pycse import regress
from nose.tools import raises
import numpy as np

T = np.array([   0,  100,  200,  300,  400,  500,  600,  700,
                 800,  900, 1000, 1100, 1200, 1300, 1400.0])

E = np.array([  6.5742 ,   6.56969,   6.64349,   6.78516,   6.85911,
                6.77201, 6.45256,   5.82957,   4.81762,   3.3017,
                1.11211,  -2.02724, -6.6282 , -13.85549, -27.67995])


def test():
    A = np.column_stack([T**0, T**1, T**2, T**3, T**4])
    p, pint, se = regress(A, E, alpha=0.05)
    print(p)
    print(pint)
    print(se)
    # there should not be any np.nan in se
    assert ~np.isnan(se).all()

test()
