import pandas as pd
from scipy.optimize import minimize
from scipy import signal, optimize
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import statsmodels.api as sm
import itertools

#-------------------------------------------------
"""
Functions to work with ARMA models @ MARCH 2017
"""
#-------------------------------------------------

###################################################################
def generateArma(pars, sz):

    gamma = pars[0]
    theta = pars[1]

    epsilon = np.random.standard_normal(sz*2)
    y = np.repeat(np.mean(epsilon), sz*2)

    for t in xrange(2, sz*2):
        y[t] = gamma * y[t - 1] + epsilon[t] + theta * epsilon[t - 1]

    return y[sz:2*sz], epsilon[sz:2*sz]


###################################################################
def fitArma(pars, data):
    gamma = pars[0]
    theta = pars[1]

    y         = np.repeat(np.mean(data), len(data))
    errorsest = geterrors(data, pars)
    sigma2    = np.maximum(pars[-1] ** 2, 1e-6)
    #sigma2 = np.sum(errorsest**2.)/len(y)-2.
    nobs      = len(errorsest)

    for t in xrange(2, len(data)):
        y[t] = gamma * y[t - 1] + data[t] + theta * data[t - 1]

    llike = 0.5 * (nobs * np.log(sigma2)
                   + np.sum((y ** 2)) / sigma2
                   + nobs * np.log(2 * np.pi))
    return llike


###################################################################
def estimateARMA11V2(pars, data):
    llk, bestP0 = DoGridSearch(data, numpts0=20, printe=True)
    optim_kwds = dict(ftol=1e-50, full_output=True)
    args = (data)

    rh, cov_x, infodict, mesg, ier = optimize.leastsq(geterrors, bestP0, args=args, **optim_kwds)

    return rh, cov_x, infodict, mesg, ier


###################################################################
def estimateARMA11NoMeanMyWayVsTheHighwaY(pars, data):
    """Return estimation bias for each parameter.
       My implementation is the left; the package is on the right
    """

    llk, bestP0 = DoGridSearch(data, numpts0=5, printe=False)
    # update
    optim_kwds = dict(ftol=1e-10, full_output=True)
    args = (data)

    # rh, cov_x, infodict, mesg, ier = optimize.leastsq(geterrors, bestP0, args=args, **optim_kwds)
    res = minimize(fitArma, bestP0, args=args, method='Nelder-Mead')
    MyPars = np.abs(res.x) - np.abs(pars)
    print(' -> estimated pars my way = %s' % res.x)

    res2 = sm.tsa.ARMA(data, (1, 1)).fit(trend='nc')
    Theirs = np.abs(res2.params) - np.abs(pars)
    print(' -> estimated pars package = %s' % res2.params)

    print(' -> Bias My way    = %.3f;\n -> The proper way = %.3f' % (np.sum(np.abs(MyPars)), np.sum(np.abs(Theirs))))


###################################################################
def geterrors(data, params):
    # copied from sandbox.tsa.arima.ARIMA
    p, q = 1, 1
    ar = np.concatenate(([1], -params[:p]))
    ma = np.concatenate(([1], params[p:p + q]))

    # lfilter_zi requires same length for ar and ma
    maxlag = 1 + max(p, q)
    armax = np.zeros(maxlag)
    armax[:p + 1] = ar
    mamax = np.zeros(maxlag)
    mamax[:q + 1] = ma

    errorsest = signal.lfilter(ar, ma, data)

    return errorsest


###################################################################
def loglike(data, params):
    """
    Loglikelihood for arma model
    """

    errorsest = geterrors(data, params)
    sigma2 = np.maximum(params[-1] ** 2, 1e-6)
    #sigma2 = np.sum(errorsest ** 2.) / len(errorsest) - 2.
    axis = 0
    nobs = len(errorsest)

    llike = -0.5 * (nobs * np.log(sigma2)
                    + np.sum(errorsest ** 2) / sigma2
                    + nobs * np.log(2 * np.pi))
    return llike


###################################################################
def DoGridSearch(sdata, numpts0=5, printe=False):
    GRID_TH = np.linspace(0.1, 0.9, numpts0 + 2)
    GRID_TH = GRID_TH[1:-1]

    GRID_OM = np.linspace(0.1, 0.9, numpts0 + 2)
    GRID_OM = GRID_OM[1:-1]

    PTS = [GRID_OM, GRID_TH]

    pars0Vec = np.array([list(x) for x in itertools.product(*PTS)])

    R = []
    for p0 in pars0Vec:
        R.append(loglike(sdata, p0))

    # entry of R where the log-likelihood is the maximum
    maxL = np.where(R == max(R))[0][0].astype(int)

    if printe is not False:
        print('optimal starting values are %s' % pars0Vec[maxL])

    return R, pars0Vec[maxL]

