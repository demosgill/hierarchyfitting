_author__ = 'demos'

from scipy.optimize import minimize
import statistics as st
import itertools as itertools

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pylppl as lp
import datetime
import math
from numpy import *
from numba import autojit

import sys
sys.path.append('/Users/demos/Documents/Python/ipy (work)/LPPLS - Sloppy/')

import sys
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/clean_codes/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/clean_strategy/')
# Mod likelihood functions
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/LPPLS - using the modified profile likelihood method for estimating the bubble status indicator/')
from arch.univariate import ConstantMean, ZeroMean, GARCH, arch_model
from scipy.optimize import fmin_slsqp
from numpy import size, log, pi, sum, diff, array, zeros, diag, dot, mat, asarray, sqrt, copy
from numpy.linalg import inv

# For parallel loops
from multiprocessing import cpu_count
try:
    CPU_COUNT = cpu_count()
except NotImplementedError:
    CPU_COUNT = 1

try:
    from joblib import Parallel, delayed
    RUN_PARALLEL = CPU_COUNT > 1
except ImportError:
    Parallel = None
    delayed = None
    RUN_PARALLEL = False

from arch.univariate import ZeroMean, GARCH

from numba import jit
from functools import partial
import numdifftools as ndt

import matplotlib as mp
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 22}

label_size = 22
mp.rcParams['xtick.labelsize'] = label_size
mp.rcParams['ytick.labelsize'] = label_size

####################################################
#                    MPL - GAMMA                   #
####################################################

def get_hessian_for_given_gamma(pars_hat, data, fixed_gamma):

    if (len(pars_hat) > 2) == True:
        # Pre calc
        small_pars_set = np.array([pars_hat[0], pars_hat[-1]]) # get omega and beta
    else:
        small_pars_set = np.array([pars_hat[0], pars_hat[-1]]) # get omega and beta

    sigma = np.repeat(np.var(data),len(data))

    ## Sensitivity analysis
    step = 1e-5 * small_pars_set
    T = np.size(data, 0)
    scores = np.zeros((T,2))

    # for the range of two nuisance parameters
    for i in xrange(2):
        h = step[i]
        delta = np.zeros(2)
        delta[i] = h

        _, logliksplus, _ = GARCH_profile(small_pars_set + delta,
                                                   data, sigma, fixed_gamma, out=True)
        _, loglikminus, _ = GARCH_profile(small_pars_set - delta,
                                                    data, sigma, fixed_gamma, out=True)
        scores[:,i] = (logliksplus - loglikminus)/(2*h)

    # Now the Hessian
    #I = scores/T
    args = (data, sigma, fixed_gamma)
    J = hessian_2sided(GARCH_profile, small_pars_set, args) # Covariance of the scores
    J = J/T
    fisher_info_matrix = mat(inv(-J))

    return fisher_info_matrix, scores


def GARCH_LM(pars, data, sigma, gamma_fix, X_hat, out=None):

    # pormenores
    omega    = pars[0] # pars
    beta     = pars[-1] # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega + gamma_fix * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks =  0.5 * (np.log(2*pi) + np.log(sigma) + eps**2/sigma)

    # Pre-calculations
    pars_step = np.array([pars[0], gamma_fix, pars[-1]])
    Fish_step, X_step = get_hessian_for_given_gamma(pars_step, data, gamma_fix)
    detI = np.abs(np.linalg.det(np.dot(X_step.T, X_step) - Fish_step))
    detS = np.abs(np.linalg.det(np.dot(X_hat.T, X_step)))

    #Objective 2
    loglik = np.sum(logliks)
    #Lm =    loglik - 1/2*(np.abs(np.linalg.det(Fish_step)))   # ORIGINAL
    Lm = (len(data)-2-2.)/2. * np.log(loglik) - np.log(detI)/2. + np.log(detS)

    if out is None:
        return Lm
    else:
        return Lm, logliks, copy(sigma)


def estimate_garch_LM_fixed_gamma(data, GARCH_LM, fix_gamma, X_hat, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0], pars[-1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_gamma, X_hat)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.001, 0.5),
              (0.5, 0.95))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_gamma])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-fix_gamma-x[1]])}
            )

    # Estimate the model
    res = minimize(GARCH_LM, pars, args=args, constraints=cons,
                   bounds = bounds, method='SLSQP',
                   callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik  # Return parameters and loglikelihood

####################################################
#                    MPL - OMEGA                   #
####################################################

# Callback variables
_callback_iter, _callback_llf = 0, 0.0,
_callback_func_count, _callback_iter_display = 0, 1

def _callback(*args):

    global _callback_iter, _callback_iter_display
    _callback_iter += 1
    disp = 'Iteration: {0:>6},   Func. Count: {1:>6.3g},   Neg. LLF: {2}'
    if _callback_iter % _callback_iter_display == 0:
        print(disp.format(_callback_iter, _callback_func_count, _callback_llf))

    return None

def constraints():
    p, o, q = 1,0,1
    k_arch = p + o + q
    # alpha[i] >0
    # alpha[i] + gamma[i] > 0 for i<=p, otherwise gamma[i]>0
    # beta[i] >0
    # sum(alpha) + 0.5 sum(gamma) + sum(beta) < 1
    a = zeros((k_arch + 2, k_arch + 1))
    for i in range(k_arch + 1):
        a[i, i] = 1.0
    for i in range(o):
        if i < p:
            a[i + p + 1, i + 1] = 1.0

    a[k_arch + 1, 1:] = -1.0
    a[k_arch + 1, p + 1:p + o + 1] = -0.5
    b = zeros(k_arch + 2)
    b[k_arch + 1] = -1.0

    return a, b


def starting_values(resids):
    p, o, q = 1, 0, 1
    power = 2.0
    alphas = [.01, .05]
    gammas = alphas
    abg = [.5, .7, .9, .98]
    abgs = list(itertools.product(*[alphas, gammas, abg]))

    target = np.mean(abs(resids) ** power)
    scale = np.mean(resids ** 2) / (target ** (2.0 / power))
    target *= (scale ** power)

    svs = []
    var_bounds = variance_bounds(resids)
    backcastt = backcast(resids)
    llfs = zeros(len(abgs))
    for i, values in enumerate(abgs):
        alpha, gamma, agb = values
        sv = (1.0 - agb) * target * ones(p + o + q + 1)
        if p > 0:
            sv[1:1 + p] = alpha / p
            agb -= alpha
        if o > 0:
            sv[1 + p:1 + p + o] = gamma / o
            agb -= gamma / 2.0
        if q > 0:
            sv[1 + p + o:1 + p + o + q] = agb / q
        svs.append(sv)
        llfs[i] = _gaussian_loglikelihood(sv, resids, backcastt,
                                          var_bounds)
    loc = np.argmax(llfs)

    return svs[loc]


def backcast(resids):

    power = 2.0
    tau = min(75, resids.shape[0])
    w = (0.94 ** arange(tau))
    w = w / sum(w)
    backcast = np.sum((abs(resids[:tau]) ** power) * w)

    return backcast


def variance_bounds(resids, power=2.0):

    nobs = resids.shape[0]

    tau = min(75, nobs)
    w = 0.94 ** arange(tau)
    w = w / sum(w)
    var_bound = np.zeros(nobs)
    initial_value = w.dot(resids[:tau] ** 2.0)

    var_bounds = np.vstack((var_bound / 1e6, var_bound * 1e6)).T
    var = resids.var()
    min_upper_bound = 1 + (resids ** 2.0).max()
    lower_bound, upper_bound = var / 1e8, 1e7 * (1 + (resids ** 2.0).max())
    var_bounds[var_bounds[:, 0] < lower_bound, 0] = lower_bound
    var_bounds[var_bounds[:, 1] < min_upper_bound, 1] = min_upper_bound
    var_bounds[var_bounds[:, 1] > upper_bound, 1] = upper_bound

    if power != 2.0:
        var_bounds **= (power / 2.0)

    return var_bounds


def _gaussian_loglikelihood(parameters, resids, backcast, var_bounds):

    sigma2 = np.zeros_like(resids)
    compute_variance(parameters, resids, sigma2, backcast, var_bounds)
    return loglikelihoood([], resids, sigma2)


def compute_variance(parameters, resids, sigma2, backcast,
                     var_bounds):
    # fresids is abs(resids) ** power
    # sresids is I(resids<0)
    power = 2.0
    fresids = abs(resids) ** power
    sresids = sign(resids)

    p, o, q = 1,0,1
    nobs = resids.shape[0]

    garch_recursion(parameters, fresids, sresids, sigma2, p, o, q, nobs,
                    backcast, var_bounds)
    inv_power = 2.0 / power
    sigma2 **= inv_power

    return sigma2


def garch_recursion(parameters, fresids, sresids, sigma2, p, o, q, nobs,
                       backcast, var_bounds):

    for t in range(nobs):
        loc = 0
        sigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * fresids[t - 1 - j]
            loc += 1
        for j in range(o):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * 0.5 * backcast
            else:
                sigma2[t] += parameters[loc] \
                             * fresids[t - 1 - j] * (sresids[t - 1 - j] < 0)
            loc += 1
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
            loc += 1
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
        elif sigma2[t] > var_bounds[t, 1]:
            if not np.isinf(sigma2[t]):
                sigma2[t] = var_bounds[t, 1] + log(sigma2[t] - var_bounds[t, 1])
            else:
                sigma2[t] = var_bounds[t, 1] + 1000

    return sigma2


def _loglikelihood(parameters, sigma2, backcast, var_bounds, resids,
                   individual=False):

    # Parse parameters

    # 1. Resids
    mp, vp, dp = _parse_parameters(parameters)

    # 2. Compute sigma2 using VolatilityModel
    sigma2 = compute_variance(parameters, resids, sigma2, backcast,
                              var_bounds)

    # 3. Compute log likelihood using Distribution
    llf = loglikelihoood(dp, resids, sigma2, individual)

    _callback_llf = -1.0 * llf
    return -1.0 * llf


def loglikelihoood(parameters, resids, sigma2, individual=False):

    lls = -0.5 * (log(2 * pi) + log(sigma2) + resids ** 2.0 / sigma2)
    if individual:
        return lls
    else:
        return sum(lls)

def linear_constraint(x, *args, **kwargs):

    return kwargs['a'].dot(x) - kwargs['b']


def constraint(a, b):

    return partial(linear_constraint, a=a, b=b)

def _parse_parameters(x):

    km, kv = 2, 1
    return x[:km], x[km:km + kv], x[km + kv:]

def set_num_parallel_threads(count=None):

    global RUN_PARALLEL, CPU_COUNT
    if count <= 1:
        RUN_PARALLEL = False
    else:
        RUN_PARALLEL = True
        CPU_COUNT = count


#########################################################
#                    FUNCTIONS
#########################################################

def simulate_2(PARS, sample_size):
    zm = ZeroMean()
    zm.volatility = GARCH(p=1, q=1)
    sim_data = zm.simulate(PARS, sample_size)

    return sim_data['data']

#########################################################

def GARCH_PP(pars, data, sigma, out=None):
    # pormenores
    omega    = pars[0] # pars
    gamma    = pars[1] # pars
    beta     = pars[2] # pars

    # pormenores
    T = np.size(data,0)

    # ret = (y - mean)
    eps = data

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks = 0.5 * (log(2 * pi) + log(sigma) + eps ** 2.0 / sigma)
    loglik = sum( logliks )
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma)

#########################################################
def estimate_garch(data, GARCH_PP, update_freq=5, disp='off'):

    # initial points
    pars = starting_values(data)

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma)

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Bounds
    bounds = ((0.01, 0.5),
              (0.01, 0.5),
              (0.5, 0.99))

    # Constrains
    cons = ({'type': 'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[2]])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-x[1]-x[2]])}
            )

    # Estimate the model
    res = minimize(GARCH_PP, pars, args=args, constraints=cons, bounds=bounds,
                   method='SLSQP', callback=_callback, options={'ftol':1e-31,'disp': False})
    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik

#########################################################
def estimate_profile_likelihood_step(data, get_profile_lik_vec=False, job_pool=None, omega=False):

    if omega == False:
        # run
        gamma_range = np.linspace(0.01, 0.15, 20) ### MUDEI AQUI !!!

        if RUN_PARALLEL:
            if job_pool is None:
                job_pool = Parallel(n_jobs=CPU_COUNT)
            res = job_pool(delayed(checking_estimation_profile_at_given_gamma)(data,
                                                                               GARCH_profile,
                                                                               gamma) for gamma in gamma_range)
        else:
            res = [checking_estimation_profile_at_given_gamma(data,GARCH_profile,
                                                              gamma) for gamma in gamma_range]

        # Save results
        _logLik = np.array([x[1] for x in res])
        first_step = pd.DataFrame(_logLik, index=gamma_range, columns=['LogLik'])
        _logLik = first_step.dropna()

        # the likelihood function normalised
        RES = (_logLik / (_logLik.min()))

        # get the best gamma
        d_gamma = RES[RES == 1.].dropna().index[0]

        # estimate the remaining paramaters with gamma fixed
        hat_pars, logLik = checking_estimation_profile_at_given_gamma(data, GARCH_profile, d_gamma)

        if  get_profile_lik_vec == False:
            return np.array([hat_pars[0], d_gamma, hat_pars[1]]), -logLik
        else:
            return np.array([hat_pars[0], d_gamma, hat_pars[1]]), -logLik, -_logLik, RES#/np.float(len(data) - 2)
    else:
        # run
        omega_range = np.linspace(0.01,0.15,20)

        if RUN_PARALLEL:
            if job_pool is None:
                job_pool = Parallel(n_jobs=4)
            res = job_pool(delayed(checking_estimation_profile_at_given_omega)(data,
                                                                               GARCH_profile_omega,
                                                                               omega) for omega in omega_range)
        else:
            res = [checking_estimation_profile_at_given_omega(data,GARCH_profile_omega,
                                                              omega) for omega in omega_range]

        # Save results
        _logLik = np.array([x[1] for x in res])
        first_step = pd.DataFrame(_logLik, index=omega_range, columns=['LogLik'])
        _logLik = first_step.dropna()

        # the likelihood function normalised
        RES = (_logLik / (_logLik.min()))

        # get the best gamma
        d_omega = RES[RES == 1.].dropna().index[0]

        # estimate the remaining paramaters with gamma fixed
        hat_pars, logLik = checking_estimation_profile_at_given_omega(data, GARCH_profile_omega, d_omega)

        if  get_profile_lik_vec == False:
            return np.array([ d_omega, hat_pars[0], hat_pars[-1]]), logLik
        else:
            return np.array([ d_omega, hat_pars[0], hat_pars[-1]]), logLik,- _logLik, RES#/np.float(len(data) - 2)

#########################################################

def checking_estimation_profile_at_given_gamma(data, GARCH_profile, fix_gamma, modified=False):

    if modified == True:
        for i in range(50):
            res, hat_pars, logLik = estimate_garch_at_fixed_gamma(data, GARCH_profile, fix_gamma, modified=True)
            if np.isnan(hat_pars[0]) == True:
                pass
            else:
                return res, hat_pars, logLik
        print('not enough to eliminate zeros')
    else:
        for i in range(500):
            hat_pars, logLik = estimate_garch_at_fixed_gamma(data, GARCH_profile, fix_gamma, modified=False)
            if np.isnan(hat_pars[0]) == True:
                pass
            else:
                return hat_pars, logLik
            break
        return np.array([np.nan, np.nan]), np.nan
        print('not enough to eliminate zeros')

#########################################################

def checking_estimation_profile_at_given_omega(data, GARCH_profile_omega, fix_omega):

    for i in range(500):
        hat_pars, logLik = estimate_garch_at_fixed_omega(data, GARCH_profile_omega, fix_omega)
        if np.isnan(hat_pars[0]) == True:
            pass
        else:
            return hat_pars, logLik
        break
    return np.array([np.nan, np.nan]), np.nan

#########################################################

def estimate_garch_at_fixed_gamma(data, GARCH_profile, fix_gamma, modified=False, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0], pars[-1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_gamma)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.001, 0.5),
              (0.5, 0.99))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_gamma])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-fix_gamma-x[1]])}
            )

    # Estimate the model
    if modified is not True:
        res = minimize(GARCH_profile, pars, args=args, constraints=cons,
                       bounds = bounds, method='SLSQP',
                       callback=_callback,
                       options={'ftol': 1e-31,'disp': False})

        # Save parameters
        hat_pars = res.x
        logLik = res.fun

        return hat_pars, logLik  # Return parameters and loglikelihood
    else:
        res = minimize(GARCH_profile, pars, args=args, constraints=cons,
                       method='SLSQP',callback=_callback,
                       options={'ftol': 1e-31,'disp': False})
        # Save parameters
        hat_pars = res.x
        logLik = res.fun

        return res, hat_pars, logLik  # Return parameters and loglikelihood

#########################################################

def estimate_garch_at_fixed_omega(data, GARCH_profile_omega, fix_omega, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[1], pars[-1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_omega)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.01, 0.5),
              (0.5, 0.99))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_omega])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-x[0]-x[1]])}
            )

    # Estimate the model
    res = minimize(GARCH_profile_omega, pars, args=args, constraints=cons,
                   bounds = bounds, method='SLSQP',
                   callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik  # Return parameters and loglikelihood

#########################################################

def GARCH_profile(pars, data, sigma, gamma_fix, out=None):
    # pormenores
    omega    = pars[0] # pars
    beta     = pars[1] # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega  + gamma_fix * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks = 0.5 * (log(2 * pi) + log(sigma) + eps ** 2.0 / sigma)
    loglik = sum(logliks)
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma)

#########################################################

def GARCH_profile_omega(pars, data, sigma, omega_fix, out=None):
    # pormenores
    omega    = omega_fix
    gamma    = pars[0] # pars
    beta     = pars[1] # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega  + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks = 0.5 * (log(2 * pi) + log(sigma) + eps ** 2.0 / sigma)
    loglik = sum(logliks)
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma)

#########################################################
###### MODIFIED PROFILE LIKELIHOOD PLEASE WORK ##########
#########################################################

def normalize_profiles(Lm, Lp, gamma_range, omega=False):

    if omega == False:
        LLp = pd.DataFrame(Lp,index=gamma_range)
        LLp = LLp/LLp.max()
        LLm = pd.DataFrame(Lm,index=gamma_range)
        LLm = LLm/LLm.max()

        return LLp, LLm

    else:
        LLp = pd.DataFrame(Lp,index=gamma_range)**-1
        LLp = LLp/LLp.max()
        LLm = pd.DataFrame(Lm,index=gamma_range)**-1
        LLm = LLm/LLm.min()

        return LLp, LLm


#########################################################

def estimate_garch_LM_fixed_omega(data, GARCH_LM_omega, fix_omega, X_hat, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[1], pars[-1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_omega, X_hat)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.001, 0.5),
              (0.5, 0.99))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_omega])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-x[0]-x[1]])}
            )

    # Estimate the model
    res = minimize(GARCH_LM_omega, pars, args=args, constraints=cons,
                   bounds = bounds, method='SLSQP',
                   callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik  # Return parameters and loglikelihood

#########################################################

def hessian_2sided(fun, theta, args):
    f = fun(theta, *args)
    h = 1e-5*np.abs(theta)
    thetah = theta + h
    h = thetah - theta
    K = size(theta,0)
    h = np.diag(h)
    fp = zeros(K)
    fm = zeros(K)
    for i in xrange(K):
        fp[i] = fun(theta+h[i], *args)
        fm[i] = fun(theta-h[i], *args)
    fpp = zeros((K,K))
    fmm = zeros((K,K))
    for i in xrange(K):
        for j in xrange(i,K):
            fpp[i,j] = fun(theta + h[i] + h[j],  *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(theta - h[i] - h[j],  *args)
            fmm[j,i] = fmm[i,j]
    hh = (diag(h))
    hh = hh.reshape((K,1))
    hh = dot(hh,hh.T)
    H = zeros((K,K))
    for i in xrange(K):
        for j in xrange(i,K):
            H[i,j] = (fpp[i,j] - fp[i] - fp[j] + f
                      + f - fm[i] - fm[j] + fmm[i,j])/hh[i,j]/2
            H[j,i] = H[i,j]

    return H

#########################################################
def get_hessian_for_given_omega(pars_hat, data, fixed_omega):

    if (len(pars_hat) > 2) == True:
        # Pre calc
        small_pars_set = np.array([pars_hat[1], pars_hat[-1]]) # get gamma and beta
    else:
        small_pars_set = np.array([pars_hat[0], pars_hat[-1]]) # get gamma and beta

    sigma = np.repeat(np.var(data),len(data))

    ## Sensitivity analysis
    step = 1e-5 * small_pars_set
    T = np.size(data, 0)
    scores = np.zeros((T,2))

    # for the range of two nuisance parameters
    for i in xrange(2):
        h = step[i]
        delta = np.zeros(2)
        delta[i] = h

        _, logliksplus, _ = GARCH_profile_omega(small_pars_set + delta,
                                                   data, sigma, fixed_omega, out=True)
        _, loglikminus, _ = GARCH_profile_omega(small_pars_set - delta,
                                                    data, sigma, fixed_omega, out=True)
        scores[:,i] = (logliksplus - loglikminus)/(2*h)

    try:
        # Now the Hessian
        #I = scores/T
        args = (data, sigma, fixed_omega)
        J = hessian_2sided(GARCH_profile_omega, small_pars_set, args) # Covariance of the scores
        J = J/T
        fisher_info_matrix = mat(inv(-J))

        return fisher_info_matrix, scores
    except:
        return np.matrix([[1, 0], [0, 1]]), scores


#########################################################
def estimate_profile_and_modified_lik(data, omega=False):

    if omega == False:
        # ESTIMATE THE LOG-PROF-LIKELIHOOD AND GET OPTIMAL PARAMETERS
        pars_hat_profile, _, Lp_n, _ = estimate_profile_likelihood_step(data, job_pool=None,
                                                                        get_profile_lik_vec=True,
                                                                        omega=False)

        # GET THE FISHER INFO MATRIX AND THE SCORES COVAR
        F_hat, X_hat = get_hessian_for_given_gamma(pars_hat_profile, data, pars_hat_profile[1])

        # ESTIMATE ITERATIVELY
        gamma_range = np.linspace(0.01, 0.15, 20)

        job_pool = Parallel(n_jobs=CPU_COUNT)
        llik = job_pool(delayed(estimate_garch_LM_fixed_gamma)(data, GARCH_LM, i, X_hat) for i in gamma_range) ## TO AQUI
        _logLik = np.array([x[1] for x in llik])

        # NORMALIZE LIKELIHOODs
        Lp_df = pd.DataFrame(Lp_n, index=gamma_range)
        Lm_df = pd.DataFrame(-_logLik, index=gamma_range)

        # GET THE NUISANCE PARAMETER LM
        try:
            best_gamma = Lm_df[Lm_df == Lm_df.max().values[0]].dropna().index[0]
        except:
            best_gamma = np.nan
        Lm_pars, _ = estimate_garch_LM_fixed_gamma(data, GARCH_LM, best_gamma, X_hat)
        Lm_pars    = np.array([Lm_pars[0], best_gamma, Lm_pars[-1]])

        return Lp_df, Lm_df, pars_hat_profile, Lm_pars

    else:
        # ESTIMATE THE LOG-PROF-LIKELIHOOD AND GET OPTIMAL PARAMETERS
        pars_hat_profile, _, Lp_n, _ = estimate_profile_likelihood_step(data,
                                                           get_profile_lik_vec=True,
                                                                        omega=True)

        # GET THE FISHER INFO MATRIX AND THE SCORES COVAR
        #F_hat, X_hat = get_jacobian_for_full_model(pars_hat_profile, data)
        F_hat, X_hat = get_hessian_for_given_omega(pars_hat_profile, data, pars_hat_profile[0])

        # ESTIMATE ITERATIVELY
        omega_range = np.linspace(0.01, 0.15, 20)

        ## UNCOMENT FOR OPTIMIZATION
        job_pool = Parallel(n_jobs=4)
        llik = job_pool(delayed(estimate_garch_LM_fixed_omega)(data, GARCH_LM_omega, i, X_hat) for i in omega_range)
        _logLik = np.array([x[1] for x in llik])

        # NORMALIZE LIKELIHOODs
        #Lp_df, Lm_df = normalize_profiles(_logLik, Lp_n, omega_range, omega=True)
        Lp_df = pd.DataFrame(Lp_n, index=omega_range)
        Lm_df = pd.DataFrame(-_logLik, index=omega_range)

        # GET THE NUISANCE PARAMETER LM
        try:
            best_omega = Lm_df[Lm_df == Lm_df.max().values[0]].dropna().index[0]
        except:
            best_omega = np.nan
        Lm_pars, _ = estimate_garch_LM_fixed_omega(data, GARCH_LM_omega, best_omega, X_hat)
        Lm_pars    = np.array([best_omega, Lm_pars[0], Lm_pars[-1]])

        return Lp_df, Lm_df, pars_hat_profile, Lm_pars

#########################################################
def GARCH_LM_omega(pars, data, sigma, omega_fix, X_hat, out=None):

    # pormenores
    gamma    = pars[0] # pars
    beta     = pars[-1] # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega_fix + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks =  0.5 * (np.log(2*pi) + np.log(sigma) + eps**2/sigma)

    # Pre-calculations
    pars_step = np.array([omega_fix, pars[0], pars[-1]])
    Fish_step, X_step = get_hessian_for_given_omega(pars_step, data, omega_fix)
    detI = np.abs(np.linalg.det(np.dot(X_step.T, X_step) - Fish_step))
    detS = np.abs(np.linalg.det(np.dot(X_hat.T, X_step)))

    #Objective 2
    loglik = np.sum(logliks)
    Lm =    (len(data)-2-2.)/2. * np.log(loglik) - np.log(detI)/2. + np.log(detS)

    if out is None:
        return Lm
    else:
        return Lm, logliks, copy(sigma)


#########################################################
######              NEW METRICS                ##########

def calc_metric_qml(pars_hat, pars_true):

    cost = []

    for i in range(len(pars_true)):
        metric = ( np.abs(pars_hat[i] - pars_true[i])/pars_true[i] )**2.
        cost.append(metric)

    return np.array([np.sum(cost)])



def calc_metric_hierarchy(mplOmegaHat, mplGammaHat, mplBetaHat, pars_true):

    cost = (((mplOmegaHat[0] - pars_true[0]) / pars_true[0])**2.)
    + (((mplGammaHat[1] - pars_true[1]) / pars_true[1])**2.)
    + (((mplBetaHat[2] - pars_true[2]) / pars_true[2])**2.)

    return np.array([cost])



def get_all_estimators(data):

    # Estimate GARCH(1,1) for given data using QML
    qmlHat, _ = estimate_garch(data, GARCH_PP)

    # Estimate GARCH(1,1) for given data using Mpl_omega
    _, _, _, mplOHat = estimate_profile_and_modified_lik(data, omega=True)

    # Estimate GARCH(1,1) for given data using Mpl_gamma
    _, _, _, mplGHat = estimate_profile_and_modified_lik(data, omega=False)

    # Estimate GARCH(1,1) for given data using Mpl_beta
    _, _, _, mplBHat = estimate_profile_and_modified_lik_beta(data)

    return qmlHat, mplOHat, mplGHat, mplBHat



def get_metrics(qmlHat, mplOHat, mplGHat, mplBHats, pars_true):

    cost_qml = calc_metric_qml(qmlHat, pars_true)

    cost_hierarchy = calc_metric_hierarchy(mplOHat, mplGHat, mplBHats, pars_true)

    return cost_qml, cost_hierarchy


def monteCarloMetric(pars_true, dataLength, N):

    _costQml = []
    _costHierarchy = []

    for n in range(N):

        data = simulate_2(pars_true, dataLength)

        # Get all estimators for given data
        qmlHat, mplOHat, mplGHat, mplBHats = get_all_estimators(data)

        # Get metric
        cost_qml, cost_hierarchy = get_metrics(qmlHat, mplOHat,
                                               mplGHat, mplBHats,
                                               pars_true)

        #append
        _costQml.append(cost_qml), _costHierarchy.append(cost_hierarchy)


    # Get mean, std and tur-it into a Data-Frame
    idx = np.str(dataLength)

    costDFQml       = pd.DataFrame(_costQml)
    costDFHierarchy = pd.DataFrame(_costHierarchy)

    costDFQml_m = pd.DataFrame(costDFQml.mean())
    costDFQml_s = pd.DataFrame(costDFQml.std())

    # Get mean, std and tur-it into a Data-Frame
    costDFHierarchy_m = pd.DataFrame(costDFHierarchy.mean())
    costDFHierarchy_s = pd.DataFrame(costDFHierarchy.std())

    # Sorting index
    costDFQml_m.index = [idx]
    costDFQml_s.index = [idx]
    costDFHierarchy_m.index = [idx]
    costDFHierarchy_s.index = [idx]

    return costDFQml_m, costDFQml_s, costDFHierarchy_m, costDFHierarchy_s


def get_H_analytically(data, pars_hat):

    """ Return H and eigenvalues (ANALYTICALLY) """

    T = np.size(data,0)
    sigma = np.repeat(np.var(data),T)
    args = (data, sigma)

    H = hessian_2sided(GARCH_PP, pars_hat, args)

    vals, vec = np.linalg.eigh(np.abs(-H))

    return H, vals, vec


def get_hess_lambda(par_hat, data):

    """ Return H and eigenvalues """

    T = np.size(data,0)
    sigma = np.repeat(np.var(data),T)

    # Getting the Hessian
    f = lambda x: GARCH_PP(x, data, sigma)
    Hfun = ndt.Hessian(f)

    # Hessian
    H = Hfun(par_hat)
    H = H/T

    # compute eigenvalues
    vals, vec = np.linalg.eigh(H)
    EDF = pd.DataFrame(vec, index=np.abs(vals / vals.max()), columns=['omega',
                                                         'gamma',
                                                         'beta'])

    EDF = EDF.sort_index(axis=0, ascending=False)

    return H, EDF


def calc_cost(lpars, pars):

    lw = lpars[0]
    lg = lpars[1]

    cost = (lw - pars[0] / pars[0])**2. + (lg - pars[1] / pars[0])**2.

    return cost


def ensemble_for_calculating_metric(pars, sampleSize):

    data = simulate_2(pars, sampleSize)

    # L
    pars_L, _ = estimate_garch(data, GARCH_PP)

    # MPL (omega)
    _, _, _, pars_Lm_omega = estimate_profile_and_modified_lik(data, omega=True)

    # MPL (gamma)
    _, _, _, Lm_pars_gamma = estimate_profile_and_modified_lik(data, omega=False)

    return pars_L, pars_Lm_omega, Lm_pars_gamma


def calc_metric_alles(L, Lmw, Lmg, pars):

    lCost   = calc_cost(L, pars)
    lmwCost = calc_cost(Lmw, pars)
    lmgCost = calc_cost(Lmg, pars)

    return lCost, lmwCost, lmgCost


def monte_carlo_metric(sampleSize, ntimes, pars):

    LCost, LmwCost, LmgCost = [], [], []
    col_name = [np.str(sampleSize)]

    # MC loop
    for mc in range(ntimes):
        l, lmw, lmg = ensemble_for_calculating_metric(pars, sampleSize)
        lCost_, lmwCost_, lmgCost_ = calc_metric_alles(l, lmw, lmg ,pars)

        LCost.append(lCost_), LmwCost.append(lmwCost_), LmgCost.append(lmgCost_)

    # make-it dataframe
    LCOst   = pd.DataFrame(LCost, columns=col_name);
    LmwCOst = pd.DataFrame(LmwCost, columns=col_name);
    LmgCOst = pd.DataFrame(LmgCost, columns=col_name)

    return LCOst, LmwCOst, LmgCOst


#########################
# MONET CARLE FUNCTIONE #
#########################

def monte_carlo_modprof(PARS, sample_size, N, omega=True):

    # PreAssing
    _Lp_pars = []
    _Lm_pars = []
    _reg_pars = []
    _minLogLik = []

    # SIMLATION
    for i in range(N):
        # SIMULATE AND ESTIMATE
        Lp_pars, Lm_pars, Lr_hat = check_estimation_modProf(PARS, sample_size, omega=omega)

        _Lp_pars.append(Lp_pars)
        _Lm_pars.append(Lm_pars)
        _reg_pars.append(Lr_hat)

    # Put it all into a dataframe
    RES_LP  = pd.DataFrame(_Lp_pars,columns=['omega','gamma','beta'])
    RES_LM  = pd.DataFrame(_Lm_pars,columns=['omega','gamma','beta'])
    RES_REG = pd.DataFrame(_reg_pars,columns=['omega','gamma','beta'])

    # output is the mean of each parameter and the standard deviation
    REG_PARS_MEAN = pd.DataFrame(RES_REG.mean(),columns=[np.str(sample_size)]).T
    REG_PARS_STD  = pd.DataFrame(RES_REG.std(),columns=[np.str(sample_size)]).T

    LP_PARS_MEAN = pd.DataFrame(RES_LP.mean(),columns=[np.str(sample_size)]).T
    LP_PARS_STD  = pd.DataFrame(RES_LP.std(),columns=[np.str(sample_size)]).T

    LM_PARS_MEAN = pd.DataFrame(RES_LM.mean(),columns=[np.str(sample_size)]).T
    LM_PARS_STD  = pd.DataFrame(RES_LM.std(),columns=[np.str(sample_size)]).T

    return LP_PARS_MEAN, LP_PARS_STD, LM_PARS_MEAN, LM_PARS_STD, REG_PARS_MEAN, REG_PARS_STD

def check_estimation_modProf(PARS, sample_size, omega=True):

    # Simulate
    data = simulate_2(PARS, sample_size)

    # Estimate via LM and LP
    _, _, Lp_hat, Lm_hat = estimate_profile_and_modified_lik(data, omega=omega)

    # estimate classical GARCH
    Lr_hat, _ = estimate_garch(data, GARCH_PP)

    return Lp_hat, Lm_hat, Lr_hat

####################################################
#                    PLOTS                         #
####################################################

def plot_likelihoods(Lp_df, Lm_df, PARS, omega=True, beta=False):

    """ Plot mod. and prof. likelihood """

    Lp_df.plot()

    if omega == True:
        plt.xlabel(r'$\omega$', fontsize=19)
        plt.legend([r'$Lp(\omega|\hat{\eta}$)'], loc='best')
        plt.ylabel(r'$L(\omega|\hat{\eta})$', fontsize=19)
        plt.axvline(Lp_df[Lp_df==Lp_df.max()].dropna().index[0], color='b', linestyle='--')

        ax = plt.twinx()
        Lm_df.plot(ax=ax, color='r')
        ax.legend([r'$Lmp(\omega|\hat{\eta}$)'], loc='best', fontsize=16)
        ax.axvline(Lm_df[Lm_df==Lm_df.max()].dropna().index[0], color='r', linestyle=':')
        ax.axvline(PARS[0], color='k')
        ax.set_yticklabels('')
        plt.tight_layout()
    elif beta == False:
        plt.xlabel(r'$\gamma$', fontsize=19)
        plt.legend([r'$Lp(\gamma|\hat{\eta}$)'], loc='best')
        plt.ylabel(r'$L(\gamma|\hat{\eta})$', fontsize=19)
        plt.axvline(Lp_df[Lp_df==Lp_df.max()].dropna().index[0], color='b', linestyle='--')

        ax = plt.twinx()
        Lm_df.plot(ax=ax, color='r')
        ax.legend([r'$Lmp(\gamma|\hat{\eta}$)'], loc='best', fontsize=16)
        ax.axvline(Lm_df[Lm_df==Lm_df.max()].dropna().index[0], color='r', linestyle=':')
        ax.axvline(PARS[1], color='k')
        ax.set_yticklabels('')
        plt.tight_layout()
    elif beta == True:
        plt.xlabel(r'$\beta$', fontsize=19)
        plt.legend([r'$Lp(\beta|\hat{\eta}$)'], loc='best', fontsize=16)
        plt.ylabel(r'$L(\beta|\hat{\eta})$', fontsize=19)
        plt.axvline(Lp_df[Lp_df==Lp_df.max()].dropna().index[0], color='b', linestyle='--')

        ax = plt.twinx()
        Lm_df.plot(ax=ax, color='r')
        ax.legend([r'$Lmp(\beta|\hat{\eta}$)'], loc='best', fontsize=16)
        ax.axvline(Lm_df[Lm_df==Lm_df.max()].dropna().index[0], color='r', linestyle=':')
        ax.axvline(PARS[-1], color='k')
        ax.set_yticklabels('')

        plt.tight_layout()


def count_it(x, num=3):
    inds = range(num)
    return pd.Series([(x==i).sum() for i in inds], index=inds)


def plot_it(df):

    f, axs = plt.subplots(1,3, figsize=(8,3))
    keys = ['iomega','igamma','ibeta']

    for key,ax in zip(keys, axs):
        x = count_it(df[key], num=len(keys))/len(df)
        x.plot(kind='barh', ax=ax)

        ax.set_xlim([0,1])
        ax.set_ylim([-0.5,len(keys)-0.5])
        ax.set_ylabel('')
        ax.set_title(key[1:])

    plt.tight_layout()


def plot_it_e(df):

    keys1 = ['eomega','egamma','ebeta']
    keys2 = ['iomega','igamma','ibeta']
    rng = 6.
    fx = 7
    f, axs = plt.subplots(2,len(keys1), figsize=(fx,5))

    for k1,k2,ax in zip(keys1, keys2, np.transpose(axs)):
        x = -np.log10(df[k1])
        x.hist(bins=np.linspace(0, rng, 100), ax=ax[1], normed=True)
        ax[1].set_xlim([0, rng])
        ax[1].set_xlim([-0.5, rng+0.5])
        ax[1].set_ylim([0, 1])

        x = pd.Series(count_it(df[k2], num=len(keys1))/len(df))
        x.plot(kind='bar', ax=ax[0])
        ax[0].set_ylim([0,1])
        ax[0].set_xlim([-0.5,len(keys1)-0.5])
        ax[0].set_xticks(range(len(keys1)))
        ax[0].set_title(k1[1:], fontsize=20)
        ax[0].set_xlabel(r'$Rank$', fontsize=20)
        ax[0].set_ylabel(r'$pdf(Rank)$', fontsize=20)

        if not(k1=='etc'):
            ax[0].set_yticklabels([])
            ax[1].set_yticklabels([])
            ax[1].set_xlabel(r'$|\mathcal{S}_d|$', fontsize=20)
            ax[1].set_ylabel(r'$pdf(S_d)$', fontsize=20)

    plt.tight_layout()
    #plt.savefig('/Users/Demos/Desktop/ranking_new.pdf')


def obtain_eigenvalue_ranking(PARS, sample_size):

    # SIMULATE AND ESTIMATE
    data = simulate_2(PARS, sample_size)
    a, _ = estimate_garch(data, GARCH_PP)

    # DATAFRAME ESTIMATED PARS
    pars_h = pd.DataFrame(a); pars_h = pars_h.T
    pars_h.columns = ['omega','gamma','beta']

    # GET HESS
    _, EDF = get_hess_lambda(a, data)

    # Adding eigenvalues and ranking system
    for x in range(len(pars_h.columns)):
        pars_h[str('e')+pars_h.columns[x]] = EDF.index.values[x]

    # Adding ranking for plotting
    rrr = {}
    result = {}

    for i in range(len(EDF.index)):
        rrr[i] = np.abs(EDF.ix[EDF.index[i]]) == np.abs(EDF.ix[EDF.index[i]]).max()
        result[i] = rrr[i][rrr[i] == True].index[0]

    for x in range(len(result.values())):
        pars_h['i'+result.values()[x]] = result.keys()[x]

    return pars_h

@jit
def monte_carlo_sloppiness(PARS, sample_size, N):

    R = pd.DataFrame()

    for i in range(N):
        r = obtain_eigenvalue_ranking(PARS, sample_size)
        R = pd.concat([R,r],axis=0)

    return R

####################################################
#                    PROF BETA                     #
####################################################

def GARCH_profile_beta(pars, data, sigma, beta_fix, out=None):
    # pormenores
    omega    = pars[0]
    gamma    = pars[1] # pars
    beta     = beta_fix # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega  + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks = 0.5 * (log(2 * pi) + log(sigma) + eps ** 2.0 / sigma)
    loglik = sum(logliks)
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma)

def estimate_garch_at_fixed_beta(data, GARCH_profile_beta, fix_beta, update_freq=5, disp='off'):

    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0], pars[1]]) # omega and gammma

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_beta)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.001, 0.5),
              (0.001, 0.5))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_beta])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-x[1]-fix_beta])}
            )

    # Estimate the model
    res = minimize(GARCH_profile_beta, pars, args=args, constraints=cons,
                   bounds = bounds, method='SLSQP',
                   callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik  # Return parameters and loglikelihood


def construct_profile_beta(data):

    # ESTIMATE ITERATIVELY
    beta_range = np.linspace(0.5, 0.95, 20)

    job_pool = Parallel(n_jobs=CPU_COUNT)
    llik = job_pool(delayed(estimate_garch_at_fixed_beta)(data, GARCH_profile_beta, i) for i in beta_range)
    _logLik = np.array([x[1] for x in llik])

    _L = pd.DataFrame(-_logLik, index=beta_range)

    # Best pars
    best_beta = _L[_L == _L.max()].dropna().index[0]

    # Get other parameters
    pars_hat, _ = estimate_garch_at_fixed_beta(data, GARCH_profile_beta, best_beta)

    return np.array([pars_hat[0], pars_hat[1], best_beta]), _L

####################################################
#                MOD PROF BETA                     #
####################################################

def GARCH_LM_beta(pars, data, sigma, beta_fix, X_hat, out=None):

    # pormenores
    omega    = pars[0]  # pars
    gamma    = pars[1]  # pars
    beta     = beta_fix # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks =  0.5 * (np.log(2*pi) + np.log(sigma) + eps**2/sigma)

    # Pre-calculations
    pars_step = np.array([pars[0], pars[1], beta_fix])
    Fish_step, X_step = get_hessian_for_given_beta(pars_step, data, beta_fix)
    detI = np.abs(np.linalg.det(np.dot(X_step.T, X_step) - Fish_step))
    detS = np.abs(np.linalg.det(np.dot(X_hat.T, X_step)))

    #Objective 2
    loglik = np.sum(logliks)
    Lm =    (len(data)-2-2.)/2. * np.log(loglik) - np.log(detI)/2. + np.log(detS)

    if out is None:
        return Lm
    else:
        return Lm, logliks, copy(sigma)


def get_hessian_for_given_beta(pars_hat, data, fixed_beta):

    if (len(pars_hat) > 2) == True:
        # Pre calc
        small_pars_set = np.array([pars_hat[0], pars_hat[1]]) # get omega and gamma
    else:
        small_pars_set = np.array([pars_hat[0], pars_hat[-1]])

    sigma = np.repeat(np.var(data),len(data))

    ## Sensitivity analysis
    step = 1e-5 * small_pars_set
    T = np.size(data, 0)
    scores = np.zeros((T,2))

    # for the range of two nuisance parameters
    for i in xrange(2):
        h = step[i]
        delta = np.zeros(2)
        delta[i] = h

        _, logliksplus, _ = GARCH_profile_beta(small_pars_set + delta,
                                                   data, sigma, fixed_beta, out=True)
        _, loglikminus, _ = GARCH_profile_beta(small_pars_set - delta,
                                                    data, sigma, fixed_beta, out=True)
        scores[:,i] = (logliksplus - loglikminus)/(2*h)

    try:
        # Now the Hessian
        args = (data, sigma, fixed_omega)
        J = hessian_2sided(GARCH_profile_beta, small_pars_set, args) # Covariance of the scores
        J = J/T
        fisher_info_matrix = mat(inv(-J))

        return fisher_info_matrix, scores
    except:
        return np.matrix([[1, 0], [0, 1]]), scores


def estimate_garch_LM_fixed_beta(data, GARCH_LM_beta, fix_beta, X_hat, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0], pars[1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_beta, X_hat)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.001, 0.5),
              (0.001, 0.5))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_beta])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-x[1]-fix_beta])}
            )

    # Estimate the model
    res = minimize(GARCH_LM_beta, pars, args=args, constraints=cons,
                   bounds = bounds, method='SLSQP',
                   callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik  # Return parameters and loglikelihood

def estimate_profile_and_modified_lik_beta(data):

    # ESTIMATE THE LOG-PROF-LIKELIHOOD AND GET OPTIMAL PARAMETERS
    pars_hat_profile, Lp_beta = construct_profile_beta(data)

    # GET THE FISHER INFO MATRIX AND THE SCORES COVAR
    F_hat, X_hat = get_hessian_for_given_beta(pars_hat_profile, data, pars_hat_profile[-1])

    # ESTIMATE ITERATIVELY
    beta_range = np.linspace(0.5, 0.95, 20)

    job_pool = Parallel(n_jobs=CPU_COUNT)
    llik = job_pool(delayed(estimate_garch_LM_fixed_beta)(data, GARCH_LM_beta, i, X_hat) for i in beta_range)
    _logLik = np.array([x[1] for x in llik])

    # NORMALIZE LIKELIHOODs
    Lp_df = pd.DataFrame(Lp_beta, index=beta_range)
    Lm_df = pd.DataFrame(-_logLik, index=beta_range)

    # GET THE NUISANCE PARAMETER LM
    try:
        best_beta = Lm_df[Lm_df == Lm_df.max().values[0]].dropna().index[0]
    except:
        best_beta = np.nan
    Lm_pars, _ = estimate_garch_LM_fixed_beta(data, GARCH_LM_beta, best_beta, X_hat)
    Lm_pars    = np.array([Lm_pars[0], Lm_pars[1], best_beta])

    return Lp_df, Lm_df, pars_hat_profile, Lm_pars


####################################################
#                MOD PROF BETA                     #
####################################################

def check_estimation_modProf_beta(PARS, sample_size):

    # Simulate
    data = simulate_2(PARS, sample_size)

    # ESIMATE LP; LM BETA AS PHI
    _, _, Lp_hat, Lm_hat = estimate_profile_and_modified_lik_beta(data)

    # estimate classical GARCH
    Lr_hat, _ = estimate_garch(data, GARCH_PP)

    return Lp_hat, Lm_hat, Lr_hat


def monte_carlo_modprof_beta(PARS, sample_size, N):

    # Pre-Assign
    _Lp_pars = []
    _Lm_pars = []
    _reg_pars = []
    _minLogLik = []

    # SIMLATION
    for i in range(N):
        # SIMULATE AND ESTIMATE
        Lp_pars, Lm_pars, Lr_hat = check_estimation_modProf_beta(PARS, sample_size)

        _Lp_pars.append(Lp_pars)
        _Lm_pars.append(Lm_pars)
        _reg_pars.append(Lr_hat)

    # Put it all into a dataframe
    RES_LP  = pd.DataFrame(_Lp_pars,columns=['omega','gamma','beta'])
    RES_LM  = pd.DataFrame(_Lm_pars,columns=['omega','gamma','beta'])
    RES_REG = pd.DataFrame(_reg_pars,columns=['omega','gamma','beta'])

    # output is the mean of each parameter and the standard deviation
    REG_PARS_MEAN = pd.DataFrame(RES_REG.mean(),columns=[np.str(sample_size)]).T
    REG_PARS_STD  = pd.DataFrame(RES_REG.std(),columns=[np.str(sample_size)]).T

    LP_PARS_MEAN = pd.DataFrame(RES_LP.mean(),columns=[np.str(sample_size)]).T
    LP_PARS_STD  = pd.DataFrame(RES_LP.std(),columns=[np.str(sample_size)]).T

    LM_PARS_MEAN = pd.DataFrame(RES_LM.mean(),columns=[np.str(sample_size)]).T
    LM_PARS_STD  = pd.DataFrame(RES_LM.std(),columns=[np.str(sample_size)]).T

    return LP_PARS_MEAN, LP_PARS_STD, LM_PARS_MEAN, LM_PARS_STD, REG_PARS_MEAN, REG_PARS_STD


####################################################
#                CONTOUR BETA                      #
####################################################

def GARCH_contourGB(pars, data, sigma, gamma_fix, beta_fix, out=None):

    # pormenores
    omega    = pars[0]
    gamma    = gamma_fix # pars
    beta     = beta_fix # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks =  0.5 * (np.log(2*pi) + np.log(sigma) + eps**2/sigma)
    loglik = np.sum(logliks)

    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma)


def estimateGB_contour(data, GARCH_contourGB, fix_gamma, fix_beta, update_freq=5, disp='off'):

    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0]]) # omega

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_gamma, fix_beta)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.01, 0.02))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_gamma])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_beta])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-fix_gamma-fix_beta])}
            )

    # Estimate the model
    res = minimize(GARCH_contourGB, pars, args=args, constraints=cons,
                   bounds = bounds, method='Nelder-Mead',
                   callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik  # Return parameters and loglikelihood


####################################################
#               TRUE HIERARCHY                     #
####################################################

def garch_hierarchical(pars, data, sigma, omega_hat, out=None):
    """
    1) beta; 2) gamma; 3) omega
    return [beta_hat, gamma_hat] for fixed omega_hat
    estimated previously for a given [beta_i, gamma_i]
    (GRID SEARCH)
    """

    # pormenores
    omega    = omega_hat
    gamma    = pars[0] # pars
    beta     = pars[1] # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega_hat + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks = 0.5 * (log(2 * pi) + log(sigma) + eps ** 2.0 / sigma)
    loglik = sum(logliks)
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma)


def garch_hierarchical_beta(pars, data, sigma, beta_hat, out=None):
    """
    1) beta; 2) gamma; 3) omega
    return [omega_hat, gamma_hat] for fixed beta_hat
    estimated previously for a given [beta_i, gamma_i]
    (GRID SEARCH)
    """

    # pormenores
    omega    = pars[0]
    gamma    = pars[1]  # pars
    beta     = beta_hat # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks = 0.5 * (log(2 * pi) + log(sigma) + eps ** 2.0 / sigma)
    loglik = sum(logliks)
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma)

def estimate_garch_at_fixed_omega_h(data, garch_hierarchical, omega_hat, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[1], pars[-1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, omega_hat)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.01, 0.5),
              (0.5, 0.99))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([omega_hat])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-x[0]-x[1]])}
            )

    res = minimize(garch_hierarchical, pars, args=args, constraints=cons,
                   method='SLSQP',callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return res, hat_pars, logLik  # Return parameters and loglikelihood


def estimate_garch_at_fixed_beta_h(data, garch_hierarchical_beta, beta_hat, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0], pars[1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, beta_hat)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.01, 0.5),
              (0.01, 0.5))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([beta_hat])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-x[0]-beta_hat])}
            )

    res = minimize(garch_hierarchical_beta, pars, args=args, constraints=cons,
                   bounds = bounds, method='SLSQP',callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return res, hat_pars, logLik  # Return parameters and loglikelihood


def get_results_hierarchical(data, PARS):

    # 1st) regular GARCH
    pars_reg, llk_reg = estimate_garch(data, GARCH_PP)

    # 2) trial (read info above)
    # Modified profile lik. OMEGA -> estimate [gamma, beta]
    _, _, _, pars_Lm_o = estimate_profile_and_modified_lik(data, omega=True)
    omega_hat = pars_Lm_o[0].copy()
    _, pars_hier_o, llk_hier_o = estimate_garch_at_fixed_omega_h(data,
                                                                      garch_hierarchical,
                                                                      omega_hat)

    # 3) Trial (read above)
    # Modified prof. lik. BETA -> estimate [omega, gamma]
    _, _, _, pars_Lm_b = estimate_profile_and_modified_lik_beta(data)
    beta_hat = pars_Lm_b[-1].copy()
    _, pars_hier_b, llk_hier_b = estimate_garch_at_fixed_beta_h(data,
                                                                     garch_hierarchical_beta,
                                                                     beta_hat)

    # PROCESS RESULTS (parameters and lieklihood)
    pars_method1 = pars_reg
    pars_method2 = np.array([omega_hat, pars_hier_o[0], pars_hier_o[1]])
    pars_method3 = np.array([pars_hier_b[0], pars_hier_b[1], beta_hat])

    llk_m1 = llk_reg
    llk_m2 = llk_hier_o
    llk_m3 = llk_hier_b

    return pars_method1, pars_method2, pars_method3, llk_m1, llk_m2, llk_m3


def monte_carlo_hierarchy(PARS, length_data, N):

    PM1, PM2, PM3, LLM1, LLM2, LLM3 = [], [], [], [], [], []

    for MC in range(N):
        # Simulate
        data = simulate_2(PARS, length_data)

        # Estimate
        pars_method1, pars_method2, pars_method3, llk_m1, llk_m2, llk_m3 = get_results_hierarchical(data,
                                                                                                     PARS)

        PM1.append(pars_method1)
        PM2.append(pars_method2)
        PM3.append(pars_method3)
        LLM1.append(llk_m1)
        LLM2.append(llk_m2)
        LLM3.append(llk_m3)

        lbls = ['omega', 'gamma', 'beta']

    return pd.DataFrame(PM1, columns=lbls), pd.DataFrame(PM2, columns=lbls), pd.DataFrame(PM3, columns=lbls), pd.DataFrame(LLM1, columns=['llk']), pd.DataFrame(LLM2, columns=['llk']), pd.DataFrame(LLM3, columns=['llk'])


########################################################################################
# ----- HIERARCHY CALIBRATION IDEA FOR HOLDING PARAMETERS FIXED AT EACH ITERATION ---- #
########################################################################################

#########################################################
def estimate_profile_and_modified_lik_II(data, beta_fixed, omega=False):

    if omega == False:
        # ESTIMATE THE LOG-PROF-LIKELIHOOD AND GET OPTIMAL PARAMETERS
        pars_hat_profile, _, Lp_n, _ = estimate_profile_likelihood_step(data, job_pool=None,
                                                                        get_profile_lik_vec=True,
                                                                        omega=False)

        # GET THE FISHER INFO MATRIX AND THE SCORES COVAR
        F_hat, X_hat = get_hessian_for_given_gamma_II(pars_hat_profile, data, pars_hat_profile[1], pars_hat_profile[-1])

        # ESTIMATE ITERATIVELY
        gamma_range = np.linspace(0.01, 0.15, 20)

        job_pool = Parallel(n_jobs=CPU_COUNT)
        llik = job_pool(delayed(estimate_garch_LM_fixed_gamma_II)(data, GARCH_LM_II, i,  beta_fixed, X_hat) for i in gamma_range) ## TO AQUI
        _logLik = np.array([x[1] for x in llik])

        # NORMALIZE LIKELIHOODs
        Lp_df = pd.DataFrame(Lp_n, index=gamma_range)
        Lm_df = pd.DataFrame(-_logLik, index=gamma_range)

        # GET THE NUISANCE PARAMETER LM
        try:
            best_gamma = Lm_df[Lm_df == Lm_df.max().values[0]].dropna().index[0]
        except:
            best_gamma = np.nan

        Lm_pars, _ = estimate_garch_LM_fixed_gamma_II(data, GARCH_LM, best_gamma, beta_fixed, X_hat)
        Lm_pars    = np.array([Lm_pars[0], best_gamma, beta_fixed])

        return Lp_df, Lm_df, pars_hat_profile, Lm_pars

    else:
        # ESTIMATE THE LOG-PROF-LIKELIHOOD AND GET OPTIMAL PARAMETERS
        pars_hat_profile, _, Lp_n, _ = estimate_profile_likelihood_step(data,
                                                           get_profile_lik_vec=True,
                                                                        omega=True)

        # GET THE FISHER INFO MATRIX AND THE SCORES COVAR
        #F_hat, X_hat = get_jacobian_for_full_model(pars_hat_profile, data)
        F_hat, X_hat = get_hessian_for_given_omega(pars_hat_profile, data, pars_hat_profile[0])

        # ESTIMATE ITERATIVELY
        omega_range = np.linspace(0.01, 0.15, 20)

        ## UNCOMENT FOR OPTIMIZATION
        job_pool = Parallel(n_jobs=4)
        llik = job_pool(delayed(estimate_garch_LM_fixed_omega)(data, GARCH_LM_omega, i, X_hat) for i in omega_range)
        _logLik = np.array([x[1] for x in llik])

        # NORMALIZE LIKELIHOODs
        #Lp_df, Lm_df = normalize_profiles(_logLik, Lp_n, omega_range, omega=True)
        Lp_df = pd.DataFrame(Lp_n, index=omega_range)
        Lm_df = pd.DataFrame(-_logLik, index=omega_range)

        # GET THE NUISANCE PARAMETER LM
        try:
            best_omega = Lm_df[Lm_df == Lm_df.max().values[0]].dropna().index[0]
        except:
            best_omega = np.nan
        Lm_pars, _ = estimate_garch_LM_fixed_omega(data, GARCH_LM_omega, best_omega, X_hat)
        Lm_pars    = np.array([best_omega, Lm_pars[0], Lm_pars[-1]])

        return Lp_df, Lm_df, pars_hat_profile, Lm_pars


def estimate_garch_LM_fixed_gamma_II(data, GARCH_LM_II, fix_gamma, fix_beta, X_hat, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_gamma, fix_beta, X_hat)

    # Bounds
    finfo = np.finfo(np.float64)
    bounds = ((0.001, 0.5),
              (0.5, 0.95))

    # 4. Estimate models using constrained optimization
    global _callback_func_count, _callback_iter, _callback_iter_display
    _callback_func_count, _callback_iter = 0, 0
    if update_freq <= 0 or disp == 'off':
        _callback_iter_display = 2 ** 31
        update_freq = 0
    else:
        _callback_iter_display = update_freq
    disp = 1 if disp == 'final' else 0

    # Constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_beta])},
           {'type':'ineq',
            'fun': lambda x: np.array([fix_gamma])},
           {'type':'ineq',
            'fun': lambda x: np.array([1-fix_gamma-fix_beta])}
            )

    # Estimate the model
    res = minimize(GARCH_LM_II, pars, args=args, constraints=cons,
                   method='SLSQP',
                   callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik  # Return parameters and loglikelihood

#########################################################
def GARCH_LM_omega_II(pars, data, sigma, omega_fix, X_hat, out=None):

    # pormenores
    gamma    = pars[0] # pars
    beta     = pars[-1] # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega_fix + gamma * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks =  0.5 * (np.log(2*pi) + np.log(sigma) + eps**2/sigma)

    # Pre-calculations
    pars_step = np.array([omega_fix, pars[0], pars[-1]])
    Fish_step, X_step = get_hessian_for_given_omega(pars_step, data, omega_fix)
    detI = np.abs(np.linalg.det(np.dot(X_step.T, X_step) - Fish_step))
    detS = np.abs(np.linalg.det(np.dot(X_hat.T, X_step)))

    #Objective 2
    loglik = np.sum(logliks)
    Lm =    (len(data)-2-2.)/2. * np.log(loglik) - np.log(detI)/2. + np.log(detS)

    if out is None:
        return Lm
    else:
        return Lm, logliks, copy(sigma)

def GARCH_LM_II(pars, data, sigma, gamma_fix, beta_fix, X_hat, out=None):

    # pormenores
    omega    = pars[0] # pars
    beta     = beta_fix # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega + gamma_fix * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks =  0.5 * (np.log(2*pi) + np.log(sigma) + eps**2/sigma)

    # Pre-calculations
    pars_step = np.array([pars[0], gamma_fix, beta_fix])
    Fish_step, X_step = get_hessian_for_given_gamma_II(pars_step, data, gamma_fix, beta_fix)
    detI = np.abs(np.linalg.det(np.dot(X_step.T, X_step) - Fish_step))
    try:
        detS = np.abs(np.linalg.det(np.dot(X_hat.T, X_step)))
    except:
        detS = 0.

    #Objective 2
    loglik = np.sum(logliks)
    #Lm =    loglik - 1/2*(np.abs(np.linalg.det(Fish_step)))   # ORIGINAL
    Lm = (len(data)-2-2.)/2. * np.log(loglik) - np.log(detI)/2. + np.log(detS)

    if out is None:
        return Lm
    else:
        return Lm, logliks, copy(sigma)

def get_hessian_for_given_gamma_II(pars_hat, data, fixed_gamma, fixed_beta):

    if (len(pars_hat) > 2) == True:
        # Pre calc
        small_pars_set = np.array([fixed_beta]) # get omega and beta
    else:
        small_pars_set = np.array([fixed_beta]) # get omega and beta

    sigma = np.repeat(np.var(data),len(data))

    ## Sensitivity analysis
    step = 1e-5 * small_pars_set
    T = np.size(data, 0)
    scores = np.zeros((T,2))

    # for the range of two nuisance parameters
    for i in xrange(1):
        h = step[i]
        delta = np.zeros(1)
        delta[i] = h

        _, logliksplus, _ = GARCH_profile_II(small_pars_set + delta,
                                                   data, sigma, fixed_gamma, fixed_beta, out=True)
        _, loglikminus, _ = GARCH_profile_II(small_pars_set - delta,
                                                    data, sigma, fixed_gamma, fixed_beta, out=True)
        scores[:,i] = (logliksplus - loglikminus)/(2*h)

    try:
        # Now the Hessian
        #I = scores/T
        args = (data, sigma, fixed_gamma, fixed_beta)
        J = hessian_2sided(GARCH_profile_II, small_pars_set, args) # Covariance of the scores
        J = J/T
        fisher_info_matrix = mat(inv(-J))

        return fisher_info_matrix, scores

    except:

        return np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan


def GARCH_profile_II(pars, data, sigma, gamma_fix, beta_fix, out=None):

    # pormenores
    omega    = pars[0] # pars
    beta     = beta_fix # pars

    # get excess return
    eps      = data
    T        = np.size(data,0)

    # The iteration
    for t in xrange(1,T):
        sigma[t] = omega  + gamma_fix * eps[t-1]**2 + beta * sigma[t-1]

    # The objective function
    logliks = 0.5 * (log(2 * pi) + log(sigma) + eps ** 2.0 / sigma)
    loglik = sum(logliks)
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma)