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

from scipy.integrate import odeint
def odelay(func, y0, xspan, event=[], **kwargs):
    '''ode wrapper with events
    func is callable, with signature func(Y, x, *args)
    y0 are the initial conditions
    tspan  is what you want to integrate over

    events is a list of callable functions with signature event(Y, t, *args).
    These functions return zero when an event has happend.

    We integrate manually each step.
    '''

    x0 = xspan[0]
    xf = xspan[-1]

    f0 = func(y0, x0)

    X = [x0]
    sol = [y0]
    
    e = [event(y0, x0)]
    events = []

    for i, x1 in enumerate(xspan[0:-2]):
        x2 = xspan[i + 1]

        f1 = sol[i]

        f2 = odeint(func, f1, [x1, x2])
        X += [x2]
        sol += [f2[-1][0]]

        # Now evaluate each event
        e += [event(sol[-1], X[-1])]

        if e[-1] * e[-2] < 0:
            # change in sign detected Event detected where the sign of
            # the event has changed. The event is between xPt = X[-2]
            # and xLt = X[-1]. run a modified bisect function to
            # narrow down to find where event = 0
            xLt = X[-1]
            fLt = sol[-1]
            eLt = e[-1]

            xPt = X[-2]
            fPt = sol[-2]
            ePt = e[-2]

            j = 0
            while j < 100:
                if np.abs(xLt - xPt) < 1e-6:
                    # we know the interval to a prescribed precision now.
                    # print 'Event found between {0} and {1}'.format(x1t, x2t)
                    print 'x = {0}, event = {1}, f = {2}'.format(xLt, eLt, fLt)
                    events += [(xLt, fLt)]
                    break # and return to integrating

                m = (ePt - eLt)/(xPt - xLt) #slope of line connecting points
                                            #bracketing zero

                #estimated x where the zero is      
                new_x = -ePt / m + xPt

                # now get the new value of the integrated solution at
                # that new x
                f  = odeint(func, fPt, [xPt, new_x])
                new_f = f[-1][-1]
                new_e = event(new_f, new_x)

                # now check event sign change
                if eLt * new_e > 0:
                    xPt = new_x
                    fPt = new_f
                    ePt = new_e
                else:
                    xLt = new_x
                    fLt = new_f
                    eLt = new_e

                j += 1


    return X, sol, events

         
            
        

    
