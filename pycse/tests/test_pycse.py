from pycse.PYCSE import *

def test_deriv_2pt():
    x = [0, 1, 2]
    y = [0, 2, 4]
    dydx = deriv(x, y)
    assert (np.array([2, 2, 2]) == dydx).all()

def test_deriv_4pt():
    x = [0, 1, 2]
    y = [0, 2, 4]
    dydx = deriv(x, y, method='four-point')
    assert (np.array([2, 2, 2]) == dydx).all()

def test_deriv_fft():
    x = [0, 1, 2]
    y = [0, 2, 4]
    dydx = deriv(x, y, method='fft')

    #This is not a real test, it only checks that the function actually runs
    #without errors. I don't know a good test for this otherwise.
    assert t


def test_BVPsh():
    """Solve y''(x) = 2
    y(0) = 0 and y(1) = 1
    The solution is y = x^2

    y'(x) = y1(x)
    y1'(x) = y''(x) = 2



    """
    def ode(Y, x):
        return [Y[1], 2]

    y0 = [0, 1]
    X, Y = BVP_sh(ode, 0, 1, 0, 1, init=1)

    assert Y[-1, 0] == 1
