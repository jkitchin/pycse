import numpy as np
from scipy.stats.distributions import  t

def regress(A, y, alpha=None):
    '''linear regression with conf intervals

    A is a matrix of function values in columns

    alpha is for the 100*(1 - alpha) confidence level
    '''

    b, res, rank, s = np.linalg.lstsq(A, y)

    bint, se = None, None

    if alpha is not None:
        # compute the confidence intervals
        n = len(y)
        k = len(b)

        sigma2 = np.sum((y - np.dot(A, b))**2) / (n - k)  # RMSE

        C = sigma2 * np.linalg.inv(np.dot(A.T, A)) # covariance matrix
        se = np.sqrt(np.diag(C)) # standard error

        sT = t.ppf(1.0 - alpha/2.0, n - k) # student T multiplier
        CI = sT * se

        bint = np.array([(beta - ci, beta + ci) for beta,ci in zip(b,CI)])

    return (b, bint, se)



def nlinfit():
    '''nonlinear regression with conf intervals'''

def odelay():
    '''ode wrapper with events'''
