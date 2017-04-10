
import pandas as pd
from scipy.optimize import minimize
from numpy import log, invert
from scipy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize

import statsmodels.api as sm
import itertools

from numpy import dot
import warnings
warnings.filterwarnings('ignore')
import numdifftools as ndt
import time
from tqdm import *

#########################################################
                                                        #
""" Functions to work with ARMA models @ MARCH 2017 """ #
                                                        #
#########################################################

algorithm = 'BFGS'

# ----------------------------------------
def generateMA(par, sz):

    beta = par
    eps = np.random.standard_normal(sz)
    y = np.repeat(np.mean(eps), sz)
    error = y.copy()

    for t in xrange(2, sz):
        error[t] = eps[t] - beta * error[t - 1]

    return error, eps


# ----------------------------------------
def estimateMA(par, data):
    beta = par
    T = len(data)
    error = np.repeat(np.mean(data), T)
    sigma2 = np.var(data)

    for t in xrange(2, sz):
        error[t] = data[t] - beta * error[t - 1]

    sse = np.sum(error ** 2.) / (2 * sigma2)
    llk = 0.5 * np.log(2. * np.pi) + 0.5 * log(sigma2) + sse

    return llk


# ----------------------------------------
def generateArma(pars, sz):

    gamma = pars[0]
    theta = pars[1]

    epsilon = np.random.standard_normal(sz)
    y = np.repeat(np.mean(epsilon), sz)

    for t in xrange(2, sz):
        y[t] = gamma * y[t - 1] + (epsilon[t] + theta * epsilon[t - 1])

    return y, epsilon


# ----------------------------------------
def estimateARMA(pars, data):
    beta, theta = pars[0], pars[1]
    T = len(data)
    error = np.repeat(np.mean(data), T)
    sigma2 = np.var(data)

    for t in xrange(2, T):
        error[t] = data[t] - beta * data[t - 1] - theta * error[t - 1]

    sse = np.sum(error ** 2.) / (2 * sigma2)
    llk = 0.5 * np.log(2. * np.pi) + 0.5 * log(sigma2) + sse

    return llk


# ----------------------------------------
def estimatorARMA(data):
    # Estimate
    bounds = ((0.01, 0.99), (0.01, 0.99))
    x0 = [0.1, 0.1]
    args = (data)
    res = minimize(estimateARMA, x0, args=args, method=algorithm)

    return res.x


# ----------------------------------------
def profileARMA_beta(pars, data, betaFix, simul=False):
    beta, theta = betaFix, pars[0]
    T = len(data)
    error = np.repeat(np.mean(data), T)
    sigma2 = np.var(data)

    for t in xrange(2, T):
        error[t] = data[t] - beta * data[t - 1] - theta * error[t - 1]

    sse = np.sum(error ** 2.) / (2 * sigma2)
    llk = 0.5 * np.log(2. * np.pi) + 0.5 * log(sigma2) + sse

    if simul == False:
        return llk
    else:
        return error



def profileARMA_theta(pars, data, thetaFix, simul=False):
    beta, theta = pars[0], thetaFix
    T = len(data)
    error = np.repeat(np.mean(data), T)
    sigma2 = np.var(data)

    for t in xrange(2, T):
        error[t] = data[t] - beta * data[t - 1] - theta * error[t - 1]

    sse = np.sum(error ** 2.) / (2 * sigma2)
    llk = 0.5 * np.log(2. * np.pi) + 0.5 * log(sigma2) + sse

    if simul == False:
        return llk
    else:
        return error


# ----------------------------------------
#            MAIN FUNC I)                #
# ----------------------------------------

def profileARMA_estimator(data, beta=True):
    """ if beta == True --> profile beta
        else: profile theta
        return LLK
    """

    # Pre-alocate
    bounds = ((0.01, 0.99))
    x0 = [0.1]
    FUN, PARS = [], []
    parRange = np.linspace(0.1, 0.9, 40)  # parameter grid

    # loop: gonna profile beta or theta ?
    if beta == True:
        fun = profileARMA_beta
    else:
        fun = profileARMA_theta

    for par in range(len(parRange)):
        args = (data, parRange[par])
        res = minimize(fun, x0, args=args, method=algorithm)
        FUN.append(res.fun)
        PARS.append(res.x)

    llk = pd.DataFrame(np.array(FUN), index=parRange)
    parsH = pd.DataFrame(PARS, index=parRange)

    # Max likelihood
    llkOptimal = llk[llk == llk.min()].dropna().index[0]

    # Optimal parameters
    optimalPars = parsH[parsH.index == llkOptimal]

    if beta == True:
        optimalPars = np.array([optimalPars.index[0], optimalPars[0].values[0]])
    else:
        optimalPars = np.array([optimalPars[0].values[0], optimalPars.index[0]])

    return llk, parsH, llkOptimal, optimalPars


# ----------------------------------------
# PLOT
# ----------------------------------------
def plotProfileLik(a, b, c, d):
    f, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(b, marker='s', alpha=0.5)
    ax.legend([r'$\hat{\eta}$'], fontsize=12, loc='lower left')
    ax.set_ylabel(r'Nuisance parameter value')
    ax.set_xlabel(r'Focal parameter value')
    ax.axhline(d[1], color='b')
    ax2 = plt.twinx()
    (-a).plot(ax=ax2, marker='s', color='r', alpha=0.5)
    ax2.set_ylabel(r'Log-Likelihood')
    ax2.legend([r'$llk$'], fontsize=12, loc='best')
    ax2.axvline(c, color='r')
    plt.tight_layout()


# ----------------------------------------
def plot_McResults(mVec, stdVec):
    f, ax = plt.subplots(1, 1, figsize=(6, 4))

    c = ['k', 'orange', 'c', 'b', 'g', 'r','m']
    mrk = ['s', 'v', 'o', '*', 'D','^','.']
    for i in range(len(mVec.columns)):
        stdPm = mVec[mVec.columns[i]] + stdVec[stdVec.columns[i]]
        m     = mVec[mVec.columns[i]].copy()

        #ax.fill_between(mVec.index, stdPm, m, color=c[i], alpha=0.2)
        m.plot(ax=ax, color=c[i], marker=mrk[i], linewidth=.5, markevery=1,alpha=0.7)
        #stdPm.plot(ax=ax, color=c[i], linestyle='--', linewidth=1.5)

    ax.legend([r'$e_1: Qml(\theta)$',
                '$e_2: L_{p}(beta)$',
                '$e_3:L_{p}(theta)$',
                '$e_4: L_{mp}(beta)$',
                '$e_5: L_{mp}(theta)$',
                '$e_6: L_{mp}(all_pars)$'],
               fontsize=15, fancybox=True, framealpha=0.5)

    for i in range(len(mVec.columns)):
        stdPm = mVec[mVec.columns[i]] + stdVec[stdVec.columns[i]]
        m = mVec[mVec.columns[i]].copy()

        ax.fill_between(mVec.index, stdPm, m, color=c[i], alpha=0.2)
        #m.plot(ax=ax, color=c[i], marker=mrk[i], linewidth=.5, markevery=1, alpha=0.7)
        stdPm.plot(ax=ax, color=c[i], linestyle='--', linewidth=.3)

    plt.ylabel(r'$\sqrt{\chi^2}$', fontsize=22)
    plt.xlabel(r'$Sample$ $size$', fontsize=22)

    a = plt.axes([0.4, 0.6, .3, .3])
    for i in range(len(mVec.columns)):
        m = mVec[mVec.columns[i]].copy()
        stdPm = mVec[mVec.columns[i]] + stdVec[stdVec.columns[i]]

        a.fill_between(mVec.index[0:3], stdPm[0:3], m[0:3], color=c[i], alpha=0.3)
        m[0:3].plot(ax=a, color=c[i], marker=mrk[i], linewidth=.5, markersize=4, markevery=1, alpha=0.5)
        stdPm[0:3].plot(ax=a, color=c[i], linestyle='--', linewidth=.3)

    plt.tight_layout()


# ----------------------------------------
# BIAS
# ----------------------------------------
def estimator_estimateBias(data, truePars, allPars=True):
    """ Aggregated parameter estimation bias -> allPars == True """

    pHarQml = estimatorARMA(data)
    _, _, _, pHatBeta    = profileARMA_estimator(data, beta=True)
    _, _, _, pHatTheta   = profileARMA_estimator(data, beta=False)
    _, _, _, pHatMlBeta  = estimate_mpl_beta(data)
    _, _, _, pHatMlTheta = estimate_mpl_theta(data)

    if allPars == True:
        biasVec = computeBias(pHarQml, pHatBeta, pHatTheta, truePars, pHatMlBeta, pHatMlTheta, allPars=True)
        return biasVec
    else:
        bvecQml, bvecLpBeta, bvecLpTheta, mplb, mplt = computeBias(pHarQml, pHatBeta, pHatTheta, truePars, pHatMlBeta, pHatMlTheta, allPars=False)
        return bvecQml, bvecLpBeta, bvecLpTheta, mplb, mplt


# ----------------------------------------
def computeBias(pHatQml, pHatBeta, pHatTheta, truePars, mplbH, mpltH, allPars=True):
    """ If allPars==True, then we compute the overall bias
        otherwise, we are interested on individual parameter bias (xi - xi^*)
    """
    absLpBB = np.abs(pHatBeta)
    absLpBT = np.abs(pHatTheta)

    absLmBB = np.abs(mplbH)
    absLmBT = np.abs(mpltH)

    qmlB = np.abs(np.abs(pHatQml) - np.abs(truePars)) / np.abs(truePars)
    lpBB = np.abs(np.abs(pHatBeta) - np.abs(truePars)) / np.abs(truePars)
    lpTB = np.abs(np.abs(pHatTheta) - np.abs(truePars)) / np.abs(truePars)
    mlBB = np.abs(np.abs(mplbH) - np.abs(truePars)) / np.abs(truePars)
    mlTB = np.abs(np.abs(mpltH) - np.abs(truePars)) / np.abs(truePars)
    # #
    LpSup_part1 = np.abs(np.abs(absLpBB[0] - np.abs(truePars[0])) / np.abs(truePars[0]))
    LpSup_part2 = np.abs(np.abs(absLpBT[-1] - np.abs(truePars[-1])) / np.abs(truePars[-1]))

    LmSup_part1 = np.abs(np.abs(absLmBB[0] - np.abs(truePars[0])) / np.abs(truePars[0]))
    LmSup_part2 = np.abs(np.abs(absLmBT[-1] - np.abs(truePars[-1])) / np.abs(truePars[-1]))

    LpSup  = LpSup_part1 + LpSup_part2
    LmSup  = LmSup_part1 + LmSup_part2

    if allPars == True:
        sum1 = np.sum(qmlB)
        DF = pd.DataFrame(np.array([sum1]), columns=['qmlBias'])
        DF['LpBetaBias']  = np.sum(lpBB)
        DF['LpThetaBias'] = np.sum(lpTB)
        DF['LmBetaBias']  = np.sum(mlBB)
        DF['LmThetaBias'] = np.sum(mlTB)
        DF['LpSup'] = LpSup
        DF['LmSup'] = LmSup

        return DF
    else:
        return qmlB, lpBB, lpTB, mlBB, mlTB


# ----------------------------------------
# Monte-Carlo
# ----------------------------------------
def MonteCarloBias(truePars, sz, MC=30):
    """ Use for single iteration """

    RES = pd.DataFrame()
    # Simulate and estimate
    for mc in range(MC):
        sdata, _ = generateArma(truePars, sz)
        biasVec = estimator_estimateBias(sdata, truePars, allPars=True)
        RES = pd.concat([RES, biasVec], axis=0)

    return RES


# ----------------------------------------
def MonteCarloBias_fullIteration(pars, MC=30):
    # For 100, 150, 200, 250 ... 1000, compute 30 Monte-Carlo simulations per sample
    MeanVec = pd.DataFrame();
    StdVec = MeanVec.copy()
    sampleGrid = np.linspace(100, 300, 10).astype(int)  # sample sizes to obtain estimates

    for sz in sampleGrid:
        print(sz)
        RES = MonteCarloBias(pars, sz, MC=MC)
        MeanVec = pd.concat([MeanVec, pd.DataFrame(RES.mean()).T], axis=0)
        StdVec = pd.concat([StdVec, pd.DataFrame(RES.std()).T], axis=0)

    # Add index
    MeanVec.index = sampleGrid;
    StdVec.index = sampleGrid

    return MeanVec, StdVec


# -------------------------------------------------------------
# Modified Profile Likelihood and Hessian estimation Functions
# -------------------------------------------------------------

def calculate_hessianMatrix(pars, data, parFix, beta=True):
    """
    :param pars: pre-estimated parameters [1x2] (beta, theta)
    :param data: ARMA(1,1)data
    :param parFix: Fixed beta (if beta == True) or fixed theta otherwise
    :param beta: Beta == True or == False (then focal parameter is theta
    :return: Negative of Hessian Matrix' Inverse
    """
    if beta == True:
        f = lambda x: profileARMA_beta(x, data, parFix)
    else:
        f = lambda x: profileARMA_theta(x, data, parFix)
    # Get Hessian
    Hfun = ndt.Hessian(f, full_output=False, method='central')
    if beta == True:
        H = Hfun(pars[-1])
    else:
        H = Hfun(pars[0])

    FIM = H**-1

    return - FIM[0][0]


# ----------------------------------------
def getFisherInfoMatrixFullARMAModel(data, pars):
    f = lambda x: estimateARMA(x, data)
    Hfun = ndt.Hessian(f, full_output=False, method='central')
    H = Hfun(pars)
    FIM = H**-1
    return - FIM


# ----------------------------------------
def calc_scoresLprofile(pars, data, parFix, beta=True):

    # passar pars com shape ->>>> ([1,1])

    ## Sensitivity analysis
    step = 1e-5 * np.array(pars)
    T = np.size(data, 0)
    scores = np.zeros((T, len(pars)))

    for i in xrange(len(pars)):
        h = step[i]
        delta = np.zeros(len(pars))
        delta[i] = h
        if beta == True:
            logliksplus = profileARMA_beta(pars[-1] + delta,
                                           data, parFix, simul=True)
            loglikminus = profileARMA_beta(pars[-1] - delta,
                                           data, parFix, simul=True)
            scores[:, i] = (logliksplus - loglikminus) / (2 * h)
        else:
            logliksplus = profileARMA_theta(pars[0] + delta,
                                            data, parFix, simul=True)
            loglikminus = profileARMA_theta(pars[0] - delta,
                                            data, parFix, simul=True)
            scores[:, i] = (logliksplus - loglikminus) / (2 * h)

    covMatrix = np.dot(scores.T, scores)**-1.
    covMatrix = covMatrix[0][0]

    return - covMatrix


# --------------------------------------------------------------
def getHandScores(pars, data, parFix, beta=True):
    H = calculate_hessianMatrix(pars, data, parFix, beta=beta)
    scores = calc_scoresLprofile(pars, data, parFix, beta=beta)
    return H, scores


# ----------------------------------------
#               ESTIMATING MPL
# ----------------------------------------
def estimate_mpl_beta(data):
    # first, Lp
    _, _, _, parsHatLp = profileARMA_estimator(data, beta=True)

    # Compute Scores and FIM
    H, Xscores = getHandScores(parsHatLp, data, parsHatLp[0], beta=True)

    # estimate the Modified Profile Likelihood
    bounds = ((0.01, 0.99))
    x0 = [0.1]
    FUN, PARS = [], []
    parRange = np.linspace(0.1, 0.9, 40)  # parameter grid

    # loop: gonna profile beta or theta ?
    fun = profileARMA_MPLbeta

    for par in range(len(parRange)):
        args = (data, parRange[par], Xscores)
        res = minimize(fun, x0, args=args, method=algorithm)
        FUN.append(res.fun)
        PARS.append(res.x)

    llk = pd.DataFrame(np.array(FUN), index=parRange)
    parsH = pd.DataFrame(PARS, index=parRange)

    # Max likelihood
    llkOptimal = llk[llk == llk.min()].dropna().index[0]

    # Optimal parameters
    optimalPars = parsH[parsH.index == llkOptimal]
    optimalPars = np.array([optimalPars.index[0], optimalPars[0].values[0]])

    return llk, parsH, llkOptimal, optimalPars


# ----------------------------------------
def estimate_mpl_theta(data):
    # first, Lp
    _, _, _, parsHatLp = profileARMA_estimator(data, beta=False)

    # Compute Scores and FIM
    H, Xscores = getHandScores(parsHatLp, data, parsHatLp[-1], beta=False)

    # estimate the Modified Profile Likelihood
    bounds = ((0.01, 0.99))
    x0 = [0.1]
    FUN, PARS = [], []
    parRange = np.linspace(0.1, 0.9, 40)  # parameter grid

    # loop: gonna profile beta or theta ?
    fun = profileARMA_MPLtheta

    for par in range(len(parRange)):
        args = (data, parRange[par], Xscores)
        res = minimize(fun, x0, args=args, method=algorithm)
        FUN.append(res.fun)
        PARS.append(res.x)

    llk = pd.DataFrame(np.array(FUN), index=parRange)
    parsH = pd.DataFrame(PARS, index=parRange)

    # Max likelihood
    llkOptimal = llk[llk == llk.min()].dropna().index[0]

    # Optimal parameters
    optimalPars = parsH[parsH.index == llkOptimal]
    optimalPars = np.array([optimalPars[0].values[0], optimalPars.index[0]])

    return llk, parsH, llkOptimal, optimalPars


# ----------------------------------------
def profileARMA_MPLbeta(pars, data, betaFix, X_hat, simul=False):

    beta, theta = betaFix, pars[0]

    T           = len(data)
    error       = np.repeat(np.mean(data), T)
    sigma2      = np.var(data)

    for t in xrange(2, T):
        error[t] = data[t] - beta * data[t - 1] - theta * error[t - 1]

    sse = np.sum(error ** 2.) / (2 * sigma2)
    llk = 0.5 * np.log(2. * np.pi) + 0.5 * log(sigma2) + sse

    ## MOD PROF
    X, H = getHandScores(pars, data, betaFix, beta=True)
    I_psi = X - H
    detI = np.abs(I_psi)
    detS = np.abs(np.dot(X_hat.T, X))

    ## The cost
    llkm = (len(data)-1-2.)/2. * np.log(llk) - np.log(detI)/2. - np.log(detS)

    if simul == False:
        return llkm
    else:
        return error


# ----------------------------------------
def profileARMA_MPLtheta(pars, data, thetaFix, X_hat, simul=False):

    beta, theta = pars[0], thetaFix

    T = len(data)
    error = np.repeat(np.mean(data), T)
    sigma2 = np.var(data)

    for t in xrange(2, T):
        error[t] = data[t] - beta * data[t - 1] - theta * error[t - 1]

    sse = np.sum(error ** 2.) / (2 * sigma2)
    llk = 0.5 * np.log(2. * np.pi) + 0.5 * log(sigma2) + sse

    ## MOD PROF
    X, H = getHandScores(pars, data, thetaFix, beta=False)
    I_psi = X - H
    detI = np.abs(I_psi)
    detS = np.abs(np.dot(X_hat.T, X))

    ## The cost
    llkm = (len(data) - 1 - 2.) / 2. * np.log(llk) - np.log(detI) / 2. - np.log(detS)

    if simul == False:
        return llkm
    else:
        return error


# ----------------------------------------
# Functions V2.
# ----------------------------------------

def fitDataWith3simpleEstimators(data):
    qml_hat = estimatorARMA(data)
    _, _, _, lpBeta_hat = profileARMA_estimator(data, beta=True)
    _, _, _, lpTheta_hat = profileARMA_estimator(data, beta=False)
    return qml_hat, lpBeta_hat, lpTheta_hat


def computeBias(data, pars):
    # Estimated parameters for data data
    e1, e2, e3 = fitDataWith3simpleEstimators(data)  # qml, lpBeta, lpTheta
    _, _, _, e5 = estimate_mpl_beta(data)  # modified Profile Beta
    _, _, _, e6 = estimate_mpl_theta(data)  # modified Profile Theta

    # bias
    E1 = np.sum((np.abs(np.abs(e1) - np.abs(pars)) / np.abs(pars)) ** 2.)
    E2 = np.sum((np.abs(np.abs(e2) - np.abs(pars)) / np.abs(pars)) ** 2.)
    E3 = np.sum((np.abs(np.abs(e3) - np.abs(pars)) / np.abs(pars)) ** 2.)
    E4 = np.sum((np.abs(np.abs(e5) - np.abs(pars)) / np.abs(pars)) ** 2.)
    E5 = np.sum((np.abs(np.abs(e6) - np.abs(pars)) / np.abs(pars)) ** 2.)

    # agg LP
    one = np.abs(np.abs(np.abs(e2[0]) - pars[0]) / pars[0]) ** 2.
    two = np.abs(np.abs(np.abs(e3[-1]) - pars[1]) / pars[1]) ** 2.
    E6 = np.abs(one + two)

    # agg MLP
    one1 = np.abs(np.abs(np.abs(e5[0]) - pars[0]) / pars[0]) ** 2.
    two1 = np.abs(np.abs(np.abs(e6[-1]) - pars[1]) / pars[1]) ** 2.
    E7 = np.abs(one1 + two1)

    return pd.DataFrame([E1, E2, E3, E4, E5, E6, E7], index=['QML', 'LPb', 'LPt',
                                                             'LMb', 'LMt',
                                                             'LPagg', 'LMagg']).T


def monteCarlo_newEstimators(pars=[0.2, 0.4], sz=100, mc=30, wantPars=False):
    # If we want the parameter values on the simulation -> False
    if wantPars is not False:
        _est1, _est2, _est3 = [], [], []
        for i in tqdm(range(mc)):
            data, _ = generateArma(pars, sz)
            est1, est2, est3 = fitDataWith3simpleEstimators(data)
            _est1.append(est1);
            _est2.append(est2);
            _est3.append(est3)
        return _est1, _est2, _est3  # RETURN PARAMETERS
    else:
        _est = pd.DataFrame()
        for i in tqdm(range(mc)):
            data, _ = generateArma(pars, sz)
            biasDF = computeBias(data, pars)
            _est = pd.concat([_est, biasDF], axis=0)
        return _est  # RETURN BIAS