# hierarchyfitting

This package provides the Modified Profile Likelihood estimator for the ARMA(1,1) and GARCH(1,1) models.
The methodology uses information geometry in order to provide more precise parameter estimates, even for simple models.


-> At the local directory type:

python setup.py install


-> Example

import ARMA_Functions as armf

import pandas as pd


# Generate synthetic time-series from an ARMA(1,1) process without constant.
##### - Simulated data from the process: $y[t] = \beta * y_{t-1} + \varepsilon_{t} + \theta * \varepsilon_{t-1}$, with $\varepsilon ~ N(0,1).$


pars = [0.2, 0.6] # parameters of the ARMA model

sz   = 300 # Sample size

data = armf.generateArma(pars, sz)

# Compare estimation bias of each one of the estimators:
##### - Quasi-max. Likelihood
##### - Profile Likelihood (beta)
##### - Profile Likelihood (theta)
##### - Modified Profile Likelihood (beta)
##### - Modified Profile Likelihood (theta)


res = armf.estimator_estimateBias(data, pars, allPars=True)

res.head()

# Conclusion:

### - The application of the MPL methodology yields more accurate estimates of $y_t$ relative to other estimators. The effect is even more pronounced for small sample sizes.
