__author__ = 'demos'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import style
style.use("ggplot")

############################################################
# FIGARCH FUNCTIONS
############################################################
def syntheticFIGARCH(pars, data, simulate=True):

    mu    = pars[0]
    omega = pars[1]
    beta  = pars[2]
    phi   = pars[3]
    d     = pars[4]

    L = .5

    # Pre-pre
    T = size(data,0)
    eps = data - mu
    sigma = np.repeat(np.var(data), T)

    # Pre-calculations
    p1 = omega*(1-beta)**-1
    p2 = 1-((1-beta)**-1) * phi * (1-L)**d

    for t in xrange(1,T):
        sigma[t] = p1*sigma[t-1]**2 + p2*eps[t-1]**2

    if simulate == True:
        y = np.sqrt(sigma)*np.random.standard_normal(len(sigma))
        return y

############################################################
def syntheticIGARCH(pars, data, simulate=True):

    mu    = pars[0]
    alpha = pars[1]
    beta  = pars[2]

    # Pre-pre
    T = size(data,0)
    eps = data - mu
    sigma = np.repeat(np.var(data), T)

    for t in xrange(1,T):
        sigma[t] = alpha + beta*(sigma[t-1]**2) + (1-beta)*(eps[t-1]**2)

    if simulate == True:
        y = np.sqrt(sigma)*np.random.standard_normal(len(sigma))
        return y

############################################################