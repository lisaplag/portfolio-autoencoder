import pandas as pd
import numpy as np
from arch.bootstrap import StationaryBootstrap
from read_data import import_data

def sharpe_ratio(x):
    mu, sigma = x.mean(), np.sqrt(x.var())
    values = np.array([x.sum(), sigma, mu / sigma ]).squeeze()
    index = ['CR', 'sigma', 'SR']
    return pd.Series(values, index=index)


R = import_data('results/Yearly_portfolio/yearly_portfolio_returns_CDAX_mp')

R1 = R.iloc[:,2]
params = sharpe_ratio(R1)


bs = StationaryBootstrap(12, R1)
results = bs.apply(sharpe_ratio, 100000)
delta_CR = results[:,0] - params[0]
delta_sigma = results[:,1] - params[1]
delta_SR = results[:,2] - params[2]

def CI(delta, q=0.95):
    delta.sort()
    abs_sorted = np.abs(delta)
    bound = abs_sorted[int(q*100000)]
    return bound

CR_bound = CI(delta_CR)
sigma_bound = CI(delta_sigma)
SR_bound = CI(delta_SR)

print(CR_bound)
print(sigma_bound)
print(SR_bound)

SR = pd.DataFrame(results[:,-1:], columns=['SR'])
CI = (np.abs(delta_SR).sort_values(by='SR')).iloc[int(0.95*100000)]
sorted = SR.sort_values(by='SR')
LB = sorted.iloc[int(0.025*100000)]
UB = sorted.iloc[int(0.975*100000)]
fig = SR.hist(bins=40)