# hierarchyfitting

This package provides the Modified Profile Likelihood estimator for the ARMA(1,1) and GARCH(1,1) models.
The methodology uses information geometry in order to provide more precise parameter estimates, even for simple models.


-> At the local directory type:

python setup.py install


-> Example

import ARMA_Functions as armf

import pandas as pd


# Generate synthetic time-series from an ARMA(1,1) process without constant.
### - Simulated data from the process: $y[t] = beta*y[t-1] + error[t] + theta*error[t-1], with error ~ N(0,1).$


pars = [0.2, 0.6] # parameters of the ARMA model

sz   = 300 # Sample size

data = armf.generateArma(pars, sz)

# Compare estimation bias of each one of the estimators:
### - Quasi-max. Likelihood
### - Profile Likelihood (beta)
### - Profile Likelihood (theta)
### - Modified Profile Likelihood (beta)
### - Modified Profile Likelihood (theta)


res = armf.estimator_estimateBias(data, pars, allPars=True)

res.head()

# Conclusion:

### - The MPL methodology yields an inferior bias relative to the other estimators.
