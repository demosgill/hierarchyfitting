__author__ = 'demos'

import htools as htools
import pandas as pd
import numpy as np
import scipy as sp
import numpy as np
import numdifftools as ndt
from scipy.optimize import minimize
import pandas.io.data as web
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA

import statsmodels.tsa.arima_process as tsp
import statsmodels.api as sm

tsa = sm.tsa

from multiprocessing import cpu_count

try:
    CPU_COUNT = 6
except NotImplementedError:
    CPU_COUNT = 1
try:
    from joblib import Parallel, delayed

    RUN_PARALLEL = CPU_COUNT > 1
except ImportError:
    Parallel = None
    delayed = None
    RUN_PARALLEL = False

from numpy import size, log, pi, sum, diff, array, zeros, diag, dot, mat, asarray, sqrt, copy
from numpy.linalg import inv
from pandas import read_csv
import pandas as pd
from scipy.optimize import fmin_slsqp
import scipy as sp
import sys
import numdifftools as ndt

import pandas as pd
import htools as htools
from scipy.optimize import minimize

import itertools
import numpy as np
import os

# ---------------------------------------------------------------
# ---------------------------------------------------------------

# DEFINE ALGO
method = 'SLSQP'

# ---------------------------------------------------------------
# ---------------------------------------------------------------
def AR(pars, data, simulate=False):
    # Parameters
    c = pars[0]
    theta = pars[1]

    # pormenores
    T = len(data)
    y = np.repeat(np.mean(data), T)
    data = data - np.mean(data)

    # function
    for t in range(T):
        y[t] = c + theta * data[t - 1]

    # objective
    sse = np.sum((data - y) ** 2.)
    sigma2 = sse / T
    llk = (T / 2 * (np.log(2 * np.pi) + np.log(sigma2)) + sse / (2 * sigma2))

    if simulate != True:
        return llk
    else:

        return y


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def MA(pars, data, simulate=False):
    # Parameters
    c = pars[0]
    theta = pars[1]

    # pormenores
    T = len(data)
    y = np.repeat(np.mean(data), T)
    data = data - np.mean(data)

    # function
    for t in range(T):
        y[t] = c + (data[t] - theta * data[t - 1])

    # objective
    sse = np.sum((data - y) ** 2.)
    sigma2 = sse / T
    llk = (T / 2 * (np.log(2 * np.pi) + np.log(sigma2)) + sse / (2 * sigma2))

    if simulate != True:
        return llk
    else:

        return y


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# noinspection PyInterpreter
def ARMA(pars, data, simulate=False):
    ## Parameters
    # EX:
    #   - const        0.002213
    #   - ar.L1.D.ORCL 0.080811
    #   - ma.L1.D.ORCL 0.182336

    c     = pars[0]
    theta = pars[1]
    phi   = pars[2]

    # pormenores
    T = len(data)
    y = np.repeat(np.mean(data), T)
    data = data  - np.mean(data)

    # function
    for t in range(T):
        y[t] = c + (y[t] - theta * y[t - 1]) + phi * data[t - 1]

    # objective
    sse = np.sum((data - y) ** 2.)
    sigma2 = sse / T
    llk = (T / 2 * (np.log(2 * np.pi) + np.log(sigma2)) + sse / (2 * sigma2))

    if simulate != True:
        return llk
    else:

        return y

# ---------------------------------------------------------------
# ---------------------------------------------------------------
def optimizeARMA(data):
    method = method
    initPars = [np.mean(data), 0.1, 0.1]
    args = (data, False)

    # Minimize
    res = minimize(ARMA, initPars, args=args, method=method)

    return res

# ---------------------------------------------------------------
# ---------------------------------------------------------------
def simulateArma(pars, sz):
    # Generate synthetic data
    epsilon = np.random.normal(0.0, 1., sz)
    syntheticData = ARMA(pars, epsilon, simulate=True)
    return syntheticData, epsilon

# ---------------------------------------------------------------
# ---------------------------------------------------------------
def MC_forGivenSampleSize(pars, sz, MC):
    S = []
    for i in range(MC):
        simData, _ = simulateArma(pars, sz)
        resSynthetic = optimizeARMA(simData)
        S.append(resSynthetic.x)

    DF = pd.DataFrame(S, columns=['p1','p2','p3'])
    return DF

# ---------------------------------------------------------------
# ---------------------------------------------------------------
def processMc(DF, sz):
    mean = pd.DataFrame(DF.mean(axis=0).values, columns=[str(sz)])
    std = pd.DataFrame(DF.std(axis=0).values, columns=[str(sz)])

    return mean.T, std.T

# ---------------------------------------------------------------
# ---------------------------------------------------------------
def MonteCarloReturningMeanAndStdAtAgivenSampleSize(pars0, sz, MC):
    DF = MC_forGivenSampleSize(pars0, sz, MC)
    m, s = processMc(DF, sz)
    return m, s

# ---------------------------------------------------------------
# ---------------------------------------------------------------
def processMCResults(results):
    M = [x[0] for x in results]
    S = [x[1] for x in results]

    MM = pd.DataFrame()
    for i in range(len(results)):
        MM = pd.concat([MM, M[i]], axis=0)

    SS = pd.DataFrame()
    for i in range(len(results)):
        SS = pd.concat([SS, S[i]], axis=0)

    return MM, SS


# ---------------------------------------------------------------
# ARMA(1) model via quasi-likelihood
def ARMA1Profile(pars, data, parFixed, par1=True, par2=False, par3=False, simulate=False):
    if par1 == True:
        mu    = parFixed
        phi   = pars[0]
        theta = pars[1]
    elif par2 == True:
        mu    = pars[0]
        phi   = parFixed
        theta = pars[1]
    elif par3 == True:
        mu    = pars[0]
        phi   = pars[1]
        theta = parFixed

    # pormenores
    T      = len(data)
    y      = np.repeat(np.mean(data), T)
    sigma2 = np.var(data)

    # distances from the mean -> r_t
    data = data

    # function
    for t in range(T):
        y[t] = mu + (y[t] - theta * y[t - 1]) + phi * data[t - 1]

    # objective
    sse = np.sum((data - y) ** 2.)
    sigma2 = sse / T
    llk = (T / 2 * (np.log(2 * np.pi) + np.log(sigma2)) + sse / (2 * sigma2))

    if simulate != True:
        return llk
    else:
        return y


# ---------------------------------------------------------------
def estimateARMA1Profile(data, parFixed, par1=False, par2=False, par3=False, costOnly=True):

    # choose par
    if par1 == True:
        args = (data, parFixed, True, False, False)
        pars0 = [0.1, 0.1]
    elif par2 == True:
        args = (data, parFixed, False, True, False)
        pars0 = [np.mean(data), 0.1]
    elif par3 == True:
        args = (data, parFixed, False, False, True)
        pars0 = [np.mean(data), 0.1]

    func = ARMA1Profile
    res1 = minimize(ARMA1Profile, pars0, args=args, method=method)
    cost = res1.fun

    if costOnly==True:
        return cost
    else:
        return res1.x # return parameters


# ---------------------------------------------------------------
def estimateLpSingePar(data, par1=True, par2=False, par3=False):
    vecP = np.linspace(-0.9, 0.9, 50)
    lp = []
    for v in vecP:
        lp.append(-estimateARMA1Profile(data, v, par1=par1, par2=par2, par3=par3))

    # get best-par
    df = pd.DataFrame(lp, index=vecP)
    lpHat = df[df == df.max()].dropna().index[0]

    # get all pars
    parsHat = estimateARMA1Profile(data, lpHat, par1=par1, par2=par2, par3=par3, costOnly=False)

    # Order parameters properly to the one we are profilling
    if par1==True:
        parsHat = np.array([lpHat, parsHat[0], parsHat[1]])
    elif par2==True:
        parsHat = np.array([parsHat[0], lpHat,  parsHat[1]])
    elif par3==True:
        parsHat = np.array([parsHat[0], parsHat[1], lpHat])

    return parsHat, lp


# ---------------------------------------------------------------
def estimateLpAllPars(data):
    # HERE WE GET THE LIKELIHOOD FUNCTION AS A VECTOR

    vecP = np.linspace(-0.9, 0.9, 50)
    lp1, lp2, lp3 = [], [], []

    for v in vecP:
        lp1.append(-estimateARMA1Profile(data, v, par1=True, par2=False, par3=False))
        lp2.append(-estimateARMA1Profile(data, v, par1=False, par2=True, par3=False))
        lp3.append(-estimateARMA1Profile(data, v, par1=False, par2=False, par3=True))

    DF = pd.DataFrame({'mu': lp1,'phi': lp2,'theta': lp3})
    DF.index = vecP

    return DF


# ---------------------------------------------------------------
#                 MODIFIED PROFILE LIKELIHOOD
# ---------------------------------------------------------------
def getHessianforGivenParARMA(pars, data, parFix, par1=True, par2=False, par3=False):
    if par1==True:
        f = lambda x: ARMA1Profile(x, data, parFix, True, False, False)
    elif par2==True:
        f = lambda x: ARMA1Profile(x, data, parFix, False, True, False)
    elif par3==True:
        f = lambda x: ARMA1Profile(x, data, parFix, False, False, True)

    try:
        # Get Hessian
        Hfun = ndt.Hessian(f)
        H = Hfun(pars)
    except:
        H = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    return H


# ---------------------------------------------------------------
def calc_scores_soe(pars, data, parFix, par1=True, par2=False, par3=False):
    if len(pars) ==2:
        pass
    else:
        print('parameters length should be = 2')

    ## Sensitivity analysis
    step = 1e-5 * pars
    T = np.size(data, 0)
    scores = np.zeros((T,len(pars)))

    for i in xrange(len(pars)):
        h = step[i]
        delta = np.zeros(len(pars))
        delta[i] = h

        logliksplus = ARMA1Profile(pars + delta,
                                   data, parFix, par1=par1, par2=par2, par3=par3, simulate=True)
        loglikminus = ARMA1Profile(pars - delta,
                                   data, parFix, par1=par1, par2=par2, par3=par3, simulate=True)
        scores[:,i] = (logliksplus - loglikminus)/(2*h)

    return scores


# ---------------------------------------------------------------
def getHandScoresGivenPar(pars, data, parFix, par1=True, par2=False, par3=False):

    H = getHessianforGivenParARMA(pars, data, parFix, par1=par1, par2=par2, par3=par3)
    S = calc_scores_soe(pars, data, parFix, par1=par1, par2=par2, par3=par3)

    return H, S


# ---------------------------------------------------------------
#                MAINS FUNCTIONS FOR THE MPL
# ---------------------------------------------------------------
def estimateModProfArmaGivenPar(data, par1=True, par2=False, par3=False, costOnly=True):

    vec = np.linspace(-0.9, 0.9, 50)

    # Getting the profile of a single parameter and getting the best-parameter
    lpHat, llk = estimateLpSingePar(data, par1=par1, par2=par2, par3=par3)

    # Get H and Scores (len(data)xn.pars)
    if par1==True:
        bpar = lpHat[0]
        H, S = getHandScoresGivenPar(lpHat[1:], data, bpar)
    elif par2==True:
        bpar = lpHat[1]
        H, S = getHandScoresGivenPar(np.array([lpHat[0], lpHat[-1]]), data, bpar)
    elif par3==True:
        bpar = lpHat[2]
        H, S = getHandScoresGivenPar(np.array([lpHat[0], lpHat[1]]), data, bpar)

    job_pool = Parallel(n_jobs=CPU_COUNT)
    llik     = job_pool(delayed(estimateMPLARMA)(data, i, S, par1=par1, par2=par2, par3=par3) for i in vec)
    #_logLik  = np.array([x[0] for x in llik])

    return llik


# ---------------------------------------------------------------
def ARMA1ProfileMPL(pars, data, parFixed, X_hat, par1=True, par2=False, par3=False, simulate=False):

    if par1 == True:
        mu    = parFixed
        phi   = pars[0]
        theta = pars[1]
    elif par2 == True:
        mu    = pars[0]
        phi   = parFixed
        theta = pars[1]
    elif par3 == True:
        mu    = pars[0]
        phi   = pars[1]
        theta = parFixed

    # pormenores
    T      = len(data)
    y      = np.repeat(np.mean(data), T)
    sigma2 = np.var(data)

    # distances from the mean -> r_t
    data = data

    # function
    for t in range(T):
        y[t] = mu + (y[t] - theta * y[t - 1]) + phi * data[t - 1]

    # objective
    sse = np.sum((data - y) ** 2.)
    sigma2 = sse / T
    llk = (T / 2 * (np.log(2 * np.pi) + np.log(sigma2)) + sse / (2 * sigma2))

    # Get details
    Fish_step, X_step = getHandScoresGivenPar(pars, data, parFixed, par1=par1, par2=par2, par3=par3)
    detI = np.abs(np.linalg.det(np.dot(X_step.T, X_step) - Fish_step))
    detS = np.abs(np.linalg.det(np.dot(X_hat.T, X_step)))

    #Objective 2
    loglik = np.sum(llk)
    Lm = (len(data)-2-2.)/2. * np.log(loglik) - np.log(detI)/2. + np.log(detS)

    if simulate != True:
        return llk
    else:
        return y


def estimateMPLARMA(data, parFixed, X_hat, par1=True, par2=False, par3=False, costOnly=True):

    # choose par
    if par1 == True:
        args = (data, parFixed, X_hat, True, False, False)
        pars0 = [0.1, 0.1]
    elif par2 == True:
        args = (data, parFixed, X_hat, False, True, False)
        pars0 = [np.mean(data), 0.1]
    elif par3 == True:
        args = (data, parFixed, X_hat, False, False, True)
        pars0 = [np.mean(data), 0.1]

    # DEFINE F()
    func = ARMA1Profile

    # MINIMIZE
    res1 = minimize(ARMA1ProfileMPL, pars0, args=args, method=method)
    cost = -res1.fun

    if par1==True:
        estimatedPars = list([parFixed]) + list(res1.x)
    elif par2==True:
        estimatedPars = list([res1.x[0]]) + list([parFixed]) + list([res1.x[1]])
    elif par3==True:
        estimatedPars = list(res1.x) + list([parFixed])

    if costOnly==True:
        return cost
    else:
        return estimatedPars # return parameters


def estimateMlpAllPars(data):

    vecP = np.linspace(-0.9, 0.9, 50) #NAO ADIANTA MUDAR AQUI: TEM DE SER NA F(estimateModProfArmaGivenPar)
    lp1, lp2, lp3 = [], [], []

    lp1 = estimateModProfArmaGivenPar(data, par1=True, par2=False, par3=False)
    lp2 = estimateModProfArmaGivenPar(data, par1=False, par2=True, par3=False)
    lp3 = estimateModProfArmaGivenPar(data, par1=False, par2=False, par3=True)

    DF = pd.DataFrame({'mu': lp1,'phi': lp2,'theta': lp3})
    DF.index = vecP

    return DF


def estimate_AllParsViaLp(data):
    lpHat1, _ = estimateLpSingePar(data, par1=True, par2=False, par3=False)
    lpHat2, _ = estimateLpSingePar(data, par1=False, par2=True, par3=False)
    lpHat3, _ = estimateLpSingePar(data, par1=False, par2=False, par3=True)

    return lpHat1, lpHat2, lpHat3


def estimate_SingleParsViaMpl(data, par1=True, par2=False, par3=False):

    vec = np.linspace(-0.9, 0.9, 50)
    llk = estimateModProfArmaGivenPar(data, par1=par1, par2=par2, par3=par3)

    # Create DF
    dfTime = pd.DataFrame(llk, index=vec)

    # Get best parameter
    bestPar = dfTime[dfTime == dfTime.max()].dropna().index[0]

    return bestPar


def retrieve_allParsMpl(data, bestPar, par1=True, par2=False, par3=False, costOnly=False):

    # Getting the profile of a single parameter and getting the best-parameter
    lpHat, llk = estimateLpSingePar(data, par1=par1, par2=par2, par3=par3)

    # Get H and Scores (len(data)xn.pars)
    if par1==True:
        bpar = lpHat[0]
        H, S = getHandScoresGivenPar(lpHat[1:], data, bpar)
    elif par2==True:
        bpar = lpHat[1]
        H, S = getHandScoresGivenPar(np.array([lpHat[0], lpHat[-1]]), data, bpar)
    elif par3==True:
        bpar = lpHat[2]
        H, S = getHandScoresGivenPar(np.array([lpHat[0], lpHat[1]]), data, bpar)

    bestPars = estimateMPLARMA(data, bestPar, S, par1=par1, par2=par2, par3=par3, costOnly=False)

    return bestPars


def estimate_full_parameterSet_mpl(data, par1=True, par2=False, par3=False):
    # First get the best parameter
    bParHat = estimate_SingleParsViaMpl(data, par1=par1, par2=par2, par3=par3)

    # Estimate remaining (nuisance) parameters via MPL
    allParsHat = retrieve_allParsMpl(data, bParHat, par1=par1, par2=par2, par3=par3, costOnly=False)

    return allParsHat


def estimate_allParametersSet_Using_Mpl(data):
    mp1 = estimate_full_parameterSet_mpl(data, par1=True, par2=False, par3=False)
    mp2 = estimate_full_parameterSet_mpl(data, par1=False, par2=True, par3=False)
    mp3 = estimate_full_parameterSet_mpl(data, par1=False, par2=False, par3=True)
    return mp1, mp2, mp3

# # # # # Adapt Monte Carlo Functions # # # # #

def MC_given_sample_lp_and_Lm(pars, sz, MC):

    L1, L2, L3 = [], [], []
    M1, M2, M3 = [], [], []

    for i in range(MC):
        # Simulate
        simData, _ = simulateArma(pars, sz)

        # Fit using LP
        l1, l2, l3 = estimate_AllParsViaLp(simData)

        # Fit using MLP
        m1, m2, m3 = estimate_allParametersSet_Using_Mpl(simData)

        # Return pars
        L1.append(l1); L2.append(l2); L3.append(l3)
        M1.append(m1); M2.append(m2); M3.append(m3)

    ML1 = pd.DataFrame(pd.DataFrame(L1).mean(axis=0),columns=[str(sz)]).T
    ML2 = pd.DataFrame(pd.DataFrame(L2).mean(axis=0),columns=[str(sz)]).T
    ML3 = pd.DataFrame(pd.DataFrame(L3).mean(axis=0),columns=[str(sz)]).T

    SL1 = pd.DataFrame(pd.DataFrame(L1).std(axis=0),columns=[str(sz)]).T
    SL2 = pd.DataFrame(pd.DataFrame(L2).std(axis=0),columns=[str(sz)]).T
    SL3 = pd.DataFrame(pd.DataFrame(L3).std(axis=0),columns=[str(sz)]).T

    MM1 = pd.DataFrame(pd.DataFrame(M1).mean(axis=0),columns=[str(sz)]).T
    MM2 = pd.DataFrame(pd.DataFrame(M2).mean(axis=0),columns=[str(sz)]).T
    MM3 = pd.DataFrame(pd.DataFrame(M3).mean(axis=0),columns=[str(sz)]).T

    SM1 = pd.DataFrame(pd.DataFrame(M1).std(axis=0),columns=[str(sz)]).T
    SM2 = pd.DataFrame(pd.DataFrame(M2).std(axis=0),columns=[str(sz)]).T
    SM3 = pd.DataFrame(pd.DataFrame(M3).std(axis=0),columns=[str(sz)]).T

    return ML1, ML2, ML3, SL1, SL2, SL3, MM1, MM2, MM3, SM1, SM2, SM3

#######################################################
def brutus_jobindex():
    return int(os.environ['LSB_JOBINDEX'])

#----------------------------------------
#----------------------------------------

def __main__():
    # Get job ID
    ID = brutus_jobindex() - 1

    # number of monte carlo per sample
    MC = 30

    # Initial Pars
    pars0 = np.array([ 0.00150058, -0.00280045, -0.1523796 ])

    # sample size vec
    sampleSize = np.arange(100, 1000, 50)

    # Saving data
    path = '/cluster/home02/mtec/gdemos/ARFIMA/'

    # samplesize
    ParForThisRun = sampleSize[ID]
    print(ID)

    # Estimator
    ML1, ML2, ML3, SL1, SL2, SL3, MM1, MM2, MM3, SM1, SM2, SM3 = MC_given_sample_lp_and_Lm(pars0, ParForThisRun, MC)

    # Saving data
    ML1.to_hdf(path+"ML1_mc30_0%i.h5" %ID, "res")
    ML2.to_hdf(path+"ML2_mc30_0%i.h5" %ID, "res")
    ML3.to_hdf(path+"ML3_mc30_0%i.h5" %ID, "res")

    SL1.to_hdf(path+"SL1_mc30_0%i.h5" %ID, "res")
    SL2.to_hdf(path+"SL2_mc30_0%i.h5" %ID, "res")
    SL3.to_hdf(path+"SL3_mc30_0%i.h5" %ID, "res")

    MM1.to_hdf(path+"MM1_mc30_0%i.h5" %ID, "res")
    MM2.to_hdf(path+"MM2_mc30_0%i.h5" %ID, "res")
    MM3.to_hdf(path+"MM3_mc30_0%i.h5" %ID, "res")

    SM1.to_hdf(path+"SM1_mc30_0%i.h5" %ID, "res")
    SM2.to_hdf(path+"SM2_mc30_0%i.h5" %ID, "res")
    SM3.to_hdf(path+"SM3_mc30_0%i.h5" %ID, "res")