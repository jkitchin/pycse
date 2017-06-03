from pycse import odelay
import numpy as np

def ode(y, x):
    return np.sin(x) * np.exp(-0.05 * x)


def minima(y, x):
    '''Approaching a minumum, dydx is negative and going to zero. our event function
is increasing

    '''
    value = ode(y, x)
    direction = 1
    isterminal = False
    return value,  isterminal, direction


def maxima(y, x):
    '''Approaching a maximum, dydx is positive and going to zero. our event function
is decreasing'''
    value = ode(y, x)
    direction = -1
    isterminal = False
    return value,  isterminal, direction

xspan = np.linspace(0.0, 20.0, 100)

y0 = 0

def test_odelay_minmax():
    X, Y, XE, YE, IE = odelay(ode, y0, xspan, events=[minima, maxima])
    assert (IE == [0, 1, 0, 1, 0, 1, 0]).all()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(X, Y)

    # blue is maximum, red is minimum
    colors = 'rb'
    for xe, ye, ie in zip(XE, YE, IE):
        plt.plot([xe], [ye], 'o', color=colors[ie])

    # plt.show()
