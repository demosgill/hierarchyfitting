__author__ = 'demos'

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import itertools
import numdifftools as ndt



# ---------------------------------------------------------------
def hubbertModel_Simulate(pars, T=100, errorLevel=0.0):

    A = pars[0]
    B = pars[1]
    C = pars[2]

    # Model
    model = np.linspace(-5., 5., T)
    grid  = np.linspace(-5., 5., T)

    # Gauss
    error = np.random.normal(0,errorLevel,T)

    for t in range(T):
        model[t] = A / (1.+np.cosh(-B*(grid[t]-C)))

    return np.array(model) + error # I AM RETURNING THE NEGATIVE OF THE DATA !!!!!


# ---------------------------------------------------------------
def hubbertModel_Fit(pars, data):

    A = pars[0]
    B = pars[1]
    C = pars[2]

    # Model
    T = len(data)
    model = np.linspace(0., 1., T)
    grid  = np.linspace(-5., 5., T)

    for t in range(T):
        model[t] = A / (1.+np.cosh(-B*(grid[t]-C)))

    sse = (model-data)**2.
    obj = np.sum(sse)

    return obj


# ---------------------------------------------------------------
def fitModel(data):
    args = (data)
    x0 = [1., .2, 0.1]
    res = minimize(hubbertModel_Fit, x0, args=args)
    return res


# ---------------------------------------------------------------
def createInitialValues(data, numpts0=2):

    GRID_A = np.linspace(0.1, 20., numpts0+2)
    GRID_B = np.linspace(0.1, 20., numpts0+2)
    GRID_C = np.linspace(0.1, 20., numpts0+2)

    def cost(x):
        return hubbertModel_Fit(x, data)
    PTS = [GRID_A, GRID_B, GRID_C]
    pars = [list(x) for x in itertools.product(*PTS)]

    return pars, cost


# ---------------------------------------------------------------
def fitNlinears(data):

    # Create initial grid
    pts0, cost = createInitialValues(data)

    res = [minimize(cost, x0, method='BFGS',
                    options={'disp': False})
           for x0 in pts0]
    objs = [r.fun for r in res]
    objs = np.array(objs)
    objs = objs[~np.isnan(objs)]
    objs = list(objs)
    mfun = min(objs)
    pars = res[objs.index(mfun)].x

    return pars, mfun


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def obtainHessian(pars, data):
    # Getting the Hessian: |profile_gammaOne(pars, data, gamma_one|
    f = lambda x: hubbertModel_Fit(x, data)

    try:
        # Get Hessian
        Hfun = ndt.Hessian(f)
        H = Hfun(pars)
    except:
        H = np.array([[0.0,0.0,0.0],
                      [0.0,0.0,0.0],
                      [0.0,0.0,0.0]])

    return H


# ---------------------------------------------------------------
def obtainJacobian(pars, data):
    # Getting the Hessian: |profile_gammaOne(pars, data, gamma_one|
    f = lambda x: hubbertModel_Fit(x, data)

    # Get Hessian
    Hfun = ndt.Jacobian(f)
    H = Hfun(pars)

    return H


# ---------------------------------------------------------------
def hubbertModel_Fit_A(pars, data, A):

    B = pars[0]
    C = pars[1]

    # Model
    T = len(data)
    model = np.linspace(0., 1., T)
    grid  = np.linspace(-5., 5., T)

    for t in range(T):
        model[t] = A / (1.+np.cosh(-B*(grid[t]-C)))

    sse = (model-data)**2.
    obj = np.sum(sse)

    return obj


# ---------------------------------------------------------------
def hubbertModel_Fit_B(pars, data, B):

    A = pars[0]
    C = pars[1]

    # Model
    T = len(data)
    model = np.linspace(0., 1., T)
    grid  = np.linspace(-5., 5., T)

    for t in range(T):
        model[t] = A / (1.+np.cosh(-B*(grid[t]-C)))

    sse = (model-data)**2.
    obj = np.sum(sse)

    return obj


# ---------------------------------------------------------------
def hubbertModel_Fit_C(pars, data, C):

    A = pars[0]
    B = pars[1]

    # Model
    T = len(data)
    model = np.linspace(0., 1., T)
    grid  = np.linspace(-5., 5., T)

    for t in range(T):
        model[t] = A / (1.+np.cosh(-B*(grid[t]-C)))

    sse = (model-data)**2.
    obj = np.sum(sse)

    return obj


# ---------------------------------------------------------------
def createInitialValues_A(data, A, numpts0=2):

    GRID_B = np.linspace(0.1, 20., numpts0+2)
    GRID_C = np.linspace(0.1, 20., numpts0+2)

    def cost(x):
        return hubbertModel_Fit_A(x, data, A)
    PTS = [GRID_B, GRID_C]
    pars = [list(x) for x in itertools.product(*PTS)]

    return pars, cost


# ---------------------------------------------------------------
def fitNlinears_A(data, A):

    # Create initial grid
    pts0, cost = createInitialValues_A(data, A)

    res = [minimize(cost, x0, method='BFGS',
                    options={'disp': False})
           for x0 in pts0]
    objs = [r.fun for r in res]
    objs = np.array(objs)
    objs = objs[~np.isnan(objs)]
    objs = list(objs)
    mfun = min(objs)
    pars = res[objs.index(mfun)].x

    return pars, mfun


# ---------------------------------------------------------------
def createInitialValues_B(data, B, numpts0=2):

    GRID_A = np.linspace(0.1, 20., numpts0+2)
    GRID_C = np.linspace(0.1, 20., numpts0+2)

    def cost(x):
        return hubbertModel_Fit_B(x, data, B)
    PTS = [GRID_A, GRID_C]
    pars = [list(x) for x in itertools.product(*PTS)]

    return pars, cost


# ---------------------------------------------------------------
def fitNlinears_B(data, B):

    # Create initial grid
    pts0, cost = createInitialValues_B(data, B)

    res = [minimize(cost, x0, method='BFGS',
                    options={'disp': False})
           for x0 in pts0]
    objs = [r.fun for r in res]
    objs = np.array(objs)
    objs = objs[~np.isnan(objs)]
    objs = list(objs)
    mfun = min(objs)
    pars = res[objs.index(mfun)].x

    return pars, mfun


# ---------------------------------------------------------------
def createInitialValues_C(data, C, numpts0=2):

    GRID_A = np.linspace(0.1, 20., numpts0+2)
    GRID_B = np.linspace(0.1, 20., numpts0+2)

    def cost(x):
        return hubbertModel_Fit_C(x, data, C)
    PTS = [GRID_A, GRID_B]
    pars = [list(x) for x in itertools.product(*PTS)]

    return pars, cost


# ---------------------------------------------------------------
def fitNlinears_C(data, C):

    # Create initial grid
    pts0, cost = createInitialValues_C(data, C)

    res = [minimize(cost, x0, method='BFGS',
                    options={'disp': False})
           for x0 in pts0]
    objs = [r.fun for r in res]
    objs = np.array(objs)
    objs = objs[~np.isnan(objs)]
    objs = list(objs)
    mfun = min(objs)
    pars = res[objs.index(mfun)].x

    return pars, mfun


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def profile_logistic(sdata, A=True, B=False, C=False, Alles=False):

    Agrid = np.linspace(0.1, 20, len(sdata))

    lp = []

    if Alles == True:
        lpA = []
        lpB = []
        lpC = []
        for i in Agrid:
            pHat, objectiveEvaluation = fitNlinears_A(sdata, i)
            lpA.append(objectiveEvaluation)

            pHat, objectiveEvaluation = fitNlinears_B(sdata, i)
            lpB.append(objectiveEvaluation)

            pHat, objectiveEvaluation = fitNlinears_C(sdata, i)
            lpC.append(objectiveEvaluation)
        return lpA, lpB, lpC
    elif A == True:
        for i in Agrid:
            pHat, objectiveEvaluation = fitNlinears_A(sdata, i)
            lp.append(objectiveEvaluation)
        parA = pd.DataFrame(lp, index=Agrid)
        return parA
    elif B == True:
        for i in Agrid:
            pHat, objectiveEvaluation = fitNlinears_A(sdata, i)
            lp.append(objectiveEvaluation)
        parB = pd.DataFrame(lp, index=Agrid)
        return parB
    elif C == True:
        for i in Agrid:
            pHat, objectiveEvaluation = fitNlinears_A(sdata, i)
            lp.append(objectiveEvaluation)
        parC = pd.DataFrame(lp, index=Agrid)
        return parC


# ---------------------------------------------------------------
def calc_scores_soe(pars, data):
    pars = np.array(pars)

    ## Sensitivity analysis
    step = 1e-5 * pars
    T = np.size(data, 0)
    scores = np.zeros((T, 3))

    for i in xrange(len(pars)):
        h = step[i]
        delta = np.zeros(3)
        delta[i] = h
        logliksplus = hubbertModel_Simulate(pars + delta, T=len(data))
        loglikminus = hubbertModel_Simulate(pars - delta, T=len(data))
        scores[:, i] = (logliksplus - loglikminus) / (2 * h)

    return scores


# ---------------------------------------------------------------
def estimate_parameters_via_LP(data, A=True, B=False, C=False):
    """
    FOR NOW USE ONLY A=True!!!!
        return all three parameters for the best A estimated via profile
        return also the profile SSE of parameter A
    """

    # 1st ---->>> construct profile
    lpA = profile_logistic(data, A=A, B=B, C=C, Alles=False)

    # Get optimal A
    A_HAT = lpA[lpA == lpA.min()].dropna().index[0]

    # Estimate remaining pars conditioned on A_HAT
    test, _ = fitNlinears_A(data, A_HAT)
    LpHats = np.array([A_HAT, test[0], test[1]])

    return LpHats, lpA


# ---------------------------------------------------------------
def modProfLikelihoodA(data):
    #### 1st ---->>> construct profile
    lpHatA, SSEProfileA = estimate_parameters_via_LP(data)

    # Calculate H and J
    X_hat = calc_scores_soe(lpHatA, data)
    #H = obtainHessian(lpHatA, data)

    ## MPL
    # Create grid and iterate
    Agrid = np.linspace(5, 15, 10)

    objMPL = []
    for A in Agrid:
        _, objectiveFunctionMpl = MPL_A_fitNlinears(data, A, X_hat)
        objMPL.append(objectiveFunctionMpl)

    return objMPL


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def MPL_A_hubbertModel_Fit(pars, data, A, X_hat):
    """
    Main function for calculating the modified profile likelihood
    inputs are:
        - pars for algorithm
        - data
        - fixed A (focal parameter); i.e. scalar
        - X_hat: score matrix calculated at the best parameters from LP
    """

    B = pars[0]
    C = pars[1]

    # Model
    T = len(data)
    model = np.linspace(0., 1., T)
    grid = np.linspace(-5., 5., T)

    for t in range(T):
        model[t] = A / (1. + np.cosh(-B * (grid[t] - C)))

    sse = (model - data) ** 2.
    sigma = np.var(sse)  # NOT SURE
    ss = np.sum(sse) / sigma

    # Likelihood function assuming gaussian errors
    logliks = 0.5 * (np.log(2 * pi) + np.log(sigma) + ss)
    loglik = np.sum(logliks)

    # Pre-calculations
    pars_step = np.array([A, pars[0], pars[-1]])
    Fish_step = obtainHessian(pars_step, data)
    X_step = calc_scores_soe(pars_step, data)
    detI = np.abs(np.linalg.det(np.dot(X_step.T, X_step) - Fish_step))
    detS = np.abs(np.linalg.det(np.dot(X_hat.T, X_step)))

    # Final Modified Profile Likelihood
    llk = (len(data) - 2 - 2.) / 2. * np.log(loglik) - np.log(detI) / 2. + np.log(detS)

    return llk


# ---------------------------------------------------------------
def MPL_A_createInitialValues(data, A, X_hat, numpts0=2):
    """
    Create starting values for the MPL_A
    """

    GRID_B = np.linspace(0.1, 20., numpts0 + 2)
    GRID_C = np.linspace(0.1, 20., numpts0 + 2)

    def cost(x):
        return MPL_A_hubbertModel_Fit(x, data, A, X_hat)

    PTS = [GRID_B, GRID_C]
    pars = [list(x) for x in itertools.product(*PTS)]

    return pars, cost


# ---------------------------------------------------------------
def MPL_A_fitNlinears(data, A, X_hat):
    # Create initial grid
    pts0, cost = MPL_A_createInitialValues(data, A, X_hat)

    res = [minimize(cost, x0, method='BFGS',
                    options={'disp': False})
           for x0 in pts0]
    objs = [r.fun for r in res]
    objs = np.array(objs)
    objs = objs[~np.isnan(objs)]
    objs = list(objs)
    mfun = min(objs)
    pars = res[objs.index(mfun)].x

    return pars, mfun