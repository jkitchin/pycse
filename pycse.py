import numpy as np
from scipy.stats.distributions import  t
from scipy.optimize import curve_fit
from scipy.integrate import odeint

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

def odelay(func, y0, xspan, events=[], TOLERANCE=1e-6, **kwargs):
    '''ode wrapper with events
    func is callable, with signature func(Y, x, *args)
    y0 are the initial conditions
    tspan  is what you want to integrate over

    events is a list of callable functions with signature event(Y, t, *args).
    These functions return zero when an event has happend.
    
    [value, isterminal, direction] = events(t,y)
    value(i) is the value of the ith event function.

    isterminal(i) = 1 if the integration is to terminate at a zero of
    this event function, otherwise, 0.

    direction(i) = 0 if all zeros are to be located (the default), +1
    if only zeros where the event function is increasing, and -1 if
    only zeros where the event function is decreasing.  
    '''

    x0 = xspan[0]  # initial point
    xf = xspan[-1] # final point

    f0 = func(y0, x0) # value of ode at initial point

    X = [x0]
    sol = [y0]
    TE, YE, IE = [], [], [] # where events occur
    
    # initial value of events
    e = np.zeros((len(events), len(xspan)))
    for i,event in enumerate(events):
        e[i,0], isterminal, direction = event(y0, x0)

    # now we step through the integration
    for i, x1 in enumerate(xspan[0:-2]):
        x2 = xspan[i + 1]
        f1 = sol[i]

        if 'full_output' in kwargs:
            f2, output = odeint(func, f1, [x1, x2], **kwargs)
            if output['message'] != 'Integration successful.':
                print output
        else:
            f2 = odeint(func, f1, [x1, x2], **kwargs)
        
        X += [x2]
        sol += [f2[-1][0]]

        # check event functions
        for j,event in enumerate(events):
            e[j, i + 1], isterminal, direction = event(sol[i + 1], X[i + 1])
    
            if e[j, i + 1] * e[j, i] < 0:
                # change in sign detected Event detected where the sign of
                # the event has changed. The event is between xPt = X[-2]
                # and xLt = X[-1]. run a modified bisect function to
                # narrow down to find where event = 0
                xLt = X[-1]
                fLt = sol[-1]
                eLt = e[j, i+1]

                xPt = X[-2]
                fPt = sol[-2]
                ePt = e[j, i]

                k = 0
                ISTERMINAL = False # assume this is the case
                
                while k < 100: # max iterations
                    if np.abs(xLt - xPt) < TOLERANCE:
                        # we know the interval to a prescribed precision now.
                        # check if direction is satisfied.
                        # e[j, i + 1] is the last value calculated
                        # e[j, i] is the previous to last
                        
                        COLLECTEVENT = False
                        # get all events
                        if direction == 0:
                            COLLECTEVENT = True
                        # only get event if event function is decreasing
                        elif (e[j, i + 1] > e[j, i] ) and direction == 1:
                            COLLECTEVENT = True
                        # only get event if event function is increasing
                        elif (e[j, i + 1] < e[j, i] ) and direction == -1:
                            COLLECTEVENT = True
                            
                        if COLLECTEVENT:
                            TE.append(xLt)
                            YE.append(fLt)
                            IE.append(j)

                            ISTERMINAL = isterminal
                        else:
                            ISTERMINAL = False
                                                    
                        break # and return to integrating

                    m = (ePt - eLt)/(xPt - xLt) #slope of line connecting points
                                                #bracketing zero

                    #estimated x where the zero is      
                    new_x = -ePt / m + xPt

                    # check if new_x is sufficiently different from xPt
                    if np.abs(new_x - xPt) < TOLERANCE:
                        # it is not different, so we do not go forward
                        xLt = new_x
                        continue                        

                    # now get the new value of the integrated solution at
                    # that new x
                    if 'full_output' in kwargs:
                        f, output  = odeint(func, fPt, [xPt, new_x], **kwargs)
                        if output['message'] != 'Integration successful.':
                            print output
                    else:
                        f  = odeint(func, fPt, [xPt, new_x], **kwargs)
                        
                    new_f = f[-1][-1]
                    new_e, isterminal, direction = event(new_f, new_x)

                    # now check event sign change
                    if eLt * new_e > 0:
                        # no sign change
                        xPt = new_x
                        fPt = new_f
                        ePt = new_e
                    else:
                        # there was a sign change
                        xLt = new_x
                        fLt = new_f
                        eLt = new_e

                    k += 1

                # if the last value of isterminal is true, break out of this loop too
                
                if ISTERMINAL:
                    # make last data point the last event
                    del X[-1], sol[-1]
                    X.append(TE[-1])
                    sol.append(YE[-1])
                    
                    return X, sol, TE, YE, IE
                
    return X, sol, TE, YE, IE





    
