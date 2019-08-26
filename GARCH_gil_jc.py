from scipy.optimize import minimize
import itertools as itertools

import pandas as pd
import numpy as np
import scipy as sp

import datetime
import math
import sys

import sys
from arch.univariate import ConstantMean, ZeroMean, GARCH, arch_model
from scipy.optimize import fmin_slsqp
	from numpy import *
from numpy.linalg import inv

from joblib import Parallel, delayed
from arch.univariate import ZeroMean, GARCH

def simulate_2(PARS, sample_size):
    zm = ZeroMean()
    zm.volatility = GARCH(p=1, q=1)
    sim_data = zm.simulate(PARS, sample_size)

    return sim_data['data']

##### GARCH QML #####

# still put in bounds param

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

def estimate_garch(data, GARCH_PP, bounds, update_freq=5, disp='off'):

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
    optpars = res.x
    logLik = res.fun

    return optpars

###### LP OMEGA ######

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

def estimate_garch_at_fixed_omega(data, GARCH_profile_omega, fix_omega, bounds, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[1], pars[-1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_omega)

    # Bounds
    finfo = np.finfo(np.float64)

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

def checking_estimation_profile_at_given_omega(data, GARCH_profile_omega, fix_omega, bounds):

    for i in range(500):
        hat_pars, logLik = estimate_garch_at_fixed_omega(data, GARCH_profile_omega, fix_omega, bounds)
        if np.isnan(hat_pars[0]) == True:
            pass
        else:
            return hat_pars, logLik
        break
    return np.array([np.nan, np.nan]), np.nan

def estimate_Lp_omega(data,bounds,omega_range,ncpus):

    job_pool = Parallel(n_jobs=ncpus)
    res = job_pool(delayed(checking_estimation_profile_at_given_omega)(data,
                                                                       GARCH_profile_omega,
                                                                       omega, bounds) for omega in omega_range)

    _logLik = np.array([x[1] for x in res])
    first_step = pd.DataFrame(_logLik, index=omega_range, columns=['LogLik'])
    _logLik = first_step.dropna()

    # the likelihood function normalised
    RES = (_logLik / (_logLik.min()))

    # get the best gamma
    d_omega = RES[RES == 1.].dropna().index[0]

    # estimate the remaining paramaters with gamma fixed
    hat_pars, logLik = checking_estimation_profile_at_given_omega(data, GARCH_profile_omega, d_omega, bounds)

    optpars = np.array([d_omega,hat_pars[0],hat_pars[-1]])

    return optpars

##### LP GAMMA #####

def GARCH_profile_gamma(pars, data, sigma, gamma_fix, out=None):
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

def estimate_garch_at_fixed_gamma(data, GARCH_profile_gamma, fix_gamma, bounds, update_freq=5, disp='off'):
    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0], pars[-1]])

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_gamma)

    # Bounds
    finfo = np.finfo(np.float64)

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

    res = minimize(GARCH_profile_gamma, pars, args=args, constraints=cons,
                   bounds = bounds, method='SLSQP',
                   callback=_callback,
                   options={'ftol': 1e-31,'disp': False})

    # Save parameters
    hat_pars = res.x
    logLik = res.fun

    return hat_pars, logLik  # Return parameters and loglikelihood

def checking_estimation_profile_at_given_gamma(data, GARCH_profile_gamma, fix_gamma, bounds):

    for i in range(500):
        hat_pars, logLik = estimate_garch_at_fixed_gamma(data, GARCH_profile_gamma, fix_gamma, bounds)
        if np.isnan(hat_pars[0]) == True:
            pass
        else:
            return hat_pars, logLik
        break
    return np.array([np.nan, np.nan]), np.nan

def estimate_Lp_gamma(data,bounds,gamma_range,ncpus):

    job_pool = Parallel(n_jobs=ncpus)
    res = job_pool(delayed(checking_estimation_profile_at_given_gamma)(data,
                                                                       GARCH_profile_gamma,
                                                                       gamma,bounds) for gamma in gamma_range)

    # Save results
    _logLik = np.array([x[1] for x in res])
    first_step = pd.DataFrame(_logLik, index=gamma_range, columns=['LogLik'])
    _logLik = first_step.dropna()

    # the likelihood function normalised
    RES = (_logLik / (_logLik.min()))

    # get the best gamma
    d_gamma = RES[RES == 1.].dropna().index[0]

    # estimate the remaining paramaters with gamma fixed
    hat_pars, logLik = checking_estimation_profile_at_given_gamma(data, GARCH_profile_gamma, d_gamma, bounds)

    optpars = np.array([hat_pars[0], d_gamma, hat_pars[1]])

    return optpars

##### LP BETA #####

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

def estimate_garch_at_fixed_beta(data, GARCH_profile_beta, fix_beta, bounds, update_freq=5, disp='off'):

    # initial points
    pars = starting_values(data)
    pars = np.array([pars[0], pars[1]]) # omega and gammma

    # Initial args
    T = np.size(data, 0)
    sigma = np.repeat(np.var(data), T)
    args = (data, sigma, fix_beta)

    # Bounds
    finfo = np.finfo(np.float64)

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

def estimate_Lp_beta(data,bounds,beta_range,ncpus):

    job_pool = Parallel(n_jobs=ncpus)
    llik = job_pool(delayed(estimate_garch_at_fixed_beta)(data,GARCH_profile_beta,i,bounds) for i in beta_range)
    _logLik = np.array([x[1] for x in llik])

    _L = pd.DataFrame(-_logLik, index=beta_range)

    # Best pars
    best_beta = _L[_L == _L.max()].dropna().index[0]

    # Get other parameters
    pars_hat, _ = estimate_garch_at_fixed_beta(data, GARCH_profile_beta, best_beta, bounds)

    optpars = np.array([pars_hat[0], pars_hat[1], best_beta])

    return optpars

##### ESTIMATE QML + ALL PROFILES #####

def get_all_profiles(data,bounds,ranges,ncpus):

    rw, rg, rb = ranges

    qml = estimate_garch(data,GARCH_PP,bounds[0])

    lpw = estimate_Lp_omega(data,bounds[1],rw,ncpus)

    lpg = estimate_Lp_gamma(data,bounds[2],rg,ncpus)

    lpb = estimate_Lp_beta(data,bounds[3],rb,ncpus)

    return qml,lpw,lpg,lpb






















############ GARCH RECURSION ##############

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

    # MSE
    target = np.mean(abs(resids) ** power)

    # Scale = 1
    scale = np.mean(resids ** 2) / (target ** (2.0 / power))
    target *= (scale ** power)
    # target > MSE

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