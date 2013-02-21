import numpy as np
from scipy.stats.distributions import  t
from scipy.optimize import curve_fit

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

def nlinfit(model, x, y, p0, alpha=0.05):
    '''nonlinear regression with conf intervals'''
    pars, pcov = curve_fit(model, x, y, p0=p0)
    n = len(y)    # number of data points
    p = len(pars) # number of parameters

    dof = max(0, n - p) # number of degrees of freedom

    # student-t value for the dof and confidence level
    tval = t.ppf(1.0-alpha/2., dof) 

    SE = []
    pint = []
    for i, p,var in zip(range(n), pars, np.diag(pcov)):
        sigma = var**0.5
        SE.append(sigma)
        pint.append([p - sigma*tval, p + sigma*tval])

    return (pars, pint, SE)

def odelay():
    '''ode wrapper with events'''
    pass
