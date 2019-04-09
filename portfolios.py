# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:46:47 2019

@author: plagl
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import read_data as data
import autoencoder as auto


def initialize_weights(num_stock):
    # initial guesses for the weights
    y0 = np.zeros(num_stock)
    for i in range(0,num_stock):
      y0[i]=1/num_stock
    y0=np.matrix(y0)
    return y0
    
    
def one_over_N(x, risk_free_asset=False):
    num_obs=len(x.index)
    in_fraction=int(0.5*num_obs)
    
    # remove column with risk-free asset returns if desired
    if risk_free_asset:
        rf = x['rf']
        rf_oos = rf.values[in_fraction:]
    else:
        rf_oos = 0
        x = x.drop(['rf'], axis = 1)
        
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    num_stock=len(x.columns)
    
    # compute means and covariance matrix
    r_avg=np.asmatrix(np.mean(x_in, axis=0))
    r_avg_oos=np.asmatrix(np.mean(x_oos, axis=0))
    sigma_oos=np.cov(x_oos,rowvar=False)
      
    # construct 1/N portfolio 
    y0 = initialize_weights(num_stock)

    # in sample performance
    returns_in=(1 + y0@r_avg.T)**252 - 1
    
    # out of sample performance
    r_portf_oos=y0@x_oos.values.T
    r_excess_oos=r_portf_oos-rf_oos
    returns_oos=(1+ y0@r_avg_oos.T)**252 - 1     
    volatility_oos=np.sqrt(252 * y0@sigma_oos@y0.T)
    sharpe_oos=returns_oos/volatility_oos
    
    print("returns in sample:", returns_in, "\nreturns out of sample:", returns_oos)
    return returns_in, returns_oos, volatility_oos, sharpe_oos, r_portf_oos, r_excess_oos
    

def mean_var_portfolio(x, risk_free_asset=True):
    num_obs=len(x.index)
    in_fraction=int(0.5*num_obs)
    
    # add column with risk-free asset returns if desired
    if risk_free_asset:
        rf = x['rf']
        rf_oos = rf.values[in_fraction:]
    else:
        rf_oos = 0
        
    # split sample   
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    num_stock=len(x.columns)
    min_ret=0.1
    
    #construct mean-variance portfolios
    y0 = initialize_weights(num_stock)
    r_avg=np.asmatrix(np.mean(x_in, axis=0))
    sigma=np.cov(x_in, rowvar=False) 
    
    # maximize returns given a certain volatility
    def objective_standard(y):
        y=np.asmatrix(y)
        return np.sum(np.sqrt(252 * y@sigma@y.T))
    def constraint1(y):
        y=np.asmatrix(y)
        return np.sum(y) - 1
    def constraint2(y):
        y=np.asmatrix(y)
        return np.sum((1 + y@r_avg.T)**252 - 1) - (min_ret)
      
    # optimize
    b = [0,1] #bounds
    bnds=[np.transpose(b)] * num_stock   #vector of b's, with length num_stock
    con1 = {'type': 'eq', 'fun': constraint1} 
    con2= {'type': 'ineq', 'fun': constraint2}
    cons = ([con1, con2])
    solution = minimize(objective_standard,y0,method='SLSQP',\
                        bounds=bnds,constraints=cons)
    weights_standard=np.asmatrix(solution.x)

    # out of sample performance
    r_portf_oos=np.matmul(weights_standard,x_oos.T)
    r_excess_oos=r_portf_oos-rf_oos
    r_excess_avg_oos=np.mean(r_excess_oos)
    r_avg_oos=np.asmatrix(np.mean(x_oos, axis=0))
    sigma_oos=np.std(r_excess_oos)
    
    returns_standard=(1 + weights_standard@r_avg_oos.T)**252 - 1 
    target=(1 + weights_standard@r_avg.T)**252 - 1

    excess_returns_standard=(1 + r_excess_avg_oos)**252 - 1
    volatility_standard=np.sqrt(252)*sigma_oos
    sharpe_standard=excess_returns_standard/volatility_standard
    
    print('In-sample return:', target)
    print('Out-of-sample return:', returns_standard)
    #print(weights_standard)
    #print(sum(weights_standard))
    return excess_returns_standard, volatility_standard, sharpe_standard, r_portf_oos, r_excess_oos

    
    
def autoencoded_portfolio(x, activation, depth, method, risk_free_asset=True):
    # split sample
    num_obs=len(x.index)
    num_stock=len(x.columns)
    in_fraction=int(0.5*num_obs)
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    min_ret=0.1
    
    # drop risk-free asset before autoencoding x_in
    x_in = x_in.drop(['rf'], axis = 1) 
   
    # compute original mean and covariance matrix
    r_avg_oos=np.asmatrix(np.mean(x_oos, axis=0))
    sigma_in=np.cov(x_in,rowvar=False)
    sigma_oos=np.cov(x_oos,rowvar=False)  

    # autoencoding and reconstructing the in-sample data
    auto_data = auto.advanced_autoencoder(x_in=x_in, epochs=50, batch_size=32, \
                                     activations=activation, depth=depth, neurons=int(num_stock/2))

    # add column with risk-free asset returns for optimization if desired
    if risk_free_asset:
        rf = x['rf']
        auto_data['rf'] = rf.values[:in_fraction]
        rf_oos = rf.values[in_fraction:]
    else:
        rf_oos = 0
        
    # set diagonal elements of autoencoded data covariance matrix equal to original variance
    if method == 'original_variance':
        auto_r_avg=np.mean(auto_data, axis=0)
        auto_sigma=np.cov(auto_data, rowvar=False)
        for i in range(0, np.size(sigma_in,1)):
            auto_sigma[i,i]=sigma_in[i,i]
    # no rescaling at all
    else:
        auto_r_avg=np.mean(auto_data, axis=0)
        auto_sigma=np.cov(auto_data, rowvar=False)
        
    auto_r_avg=np.asmatrix(auto_r_avg)
    y0 = initialize_weights(num_stock)
    
    # minimize volatility given target return
    def objective_auto(y):
        y=np.asmatrix(y)
        return np.sum(np.sqrt(252 * y@auto_sigma@y.T)) 
    def constraint1(y):
        y=np.asmatrix(y)
        return np.sum(y)-1
    def constraint2_auto(y):
        y=np.asmatrix(y)
        return np.sum((1+ y@auto_r_avg.T )**252 - 1) - (min_ret)
    
    # optimize
    b = [0,1] #bounds
    bnds=[np.transpose(b)] * num_stock   #vector of b's, with length num_stock
    con1 = {'type': 'eq', 'fun': constraint1} 
    con2_auto= {'type': 'ineq', 'fun': constraint2_auto}
    cons_auto = ([con1, con2_auto])
    solution_auto = minimize(objective_auto,y0,method='SLSQP',\
                             bounds=bnds,constraints=cons_auto)
    weights_auto=np.asmatrix(solution_auto.x)
    
    # out of sample performance
    r_portf_oos=weights_auto@x_oos.values.T
    r_excess_oos=r_portf_oos-rf_oos
    r_excess_avg_oos=np.mean(r_excess_oos)
    r_avg_oos=np.asmatrix(np.mean(x_oos.values, axis=0))
    sigma_oos=np.std(r_excess_oos)
    
    returns_auto=(1 + weights_auto@r_avg_oos.T)**252 - 1 
    target=(1 + weights_auto@auto_r_avg.T)**252 - 1

    excess_returns_auto=(1 + r_excess_avg_oos)**252 - 1
    volatility_auto=np.sqrt(252)*sigma_oos
    sharpe_auto=excess_returns_auto/volatility_auto
    
    print('In-sample return:', target)
    print('Out-of-sample return:', returns_auto)
    #print(weights_auto)
    #print(sum(weights_auto))
    return excess_returns_auto, volatility_auto, sharpe_auto, auto_data, r_portf_oos, r_excess_oos
    
      

def run(x, index, num_trials=1, write=False):
    # construct 1/N portfolio
    return_in, return_oos, volatility_oos, sharpe_oos, r_portf_n, r_excess_n = one_over_N(x)
    
    # construct standard mean-variance portfolio
    return_s, volatility_s, sharpe_s, r_portf_s, r_excess_s = mean_var_portfolio(x)
    
    # construct portfolios based on autoencoded returns     
    if num_trials == 1:
        returns_a, volatility_a, sharpe_a, auto_data, r_portf_a, r_excess_a = autoencoded_portfolio(x, 'lrelu', 3, 'original_variance')
    else:
        returns_a = np.zeros(num_trials)
        volatility_a = np.zeros(num_trials)
        sharpe_a = np.zeros(num_trials)
        for n in range(0, num_trials):
            returns_auto, volatility_auto, sharpe_auto, auto_data, r_portf_a, r_excess_a = autoencoded_portfolio(x, 'lrelu', 3, 'original_variance')
            returns_a[n], volatility_a[n], sharpe_a[n] = returns_auto, volatility_auto, sharpe_auto

    
    print("\nreturns 1/N:", return_oos, "\nvolatility 1/N:", volatility_oos, "\nsharpe 1/N:", sharpe_oos)
    print("\nreturns standard:", return_s, "\nvolatility standard:", volatility_s, "\nsharpe standard:", sharpe_s)
    print("\nreturns auto:", returns_a, "\nvolatility auto:", volatility_a, "\nsharpe auto:", sharpe_a)
    
    if write:
        r_excess = np.concatenate((r_excess_n.T, r_excess_s.T, r_excess_a.T),axis=1)
        r_excess = pd.DataFrame(r_excess, index=x[int(0.5*len(x.index)):].index, columns=['1/N', 'Standard', 'Autoencoder'])

        r_portf = np.concatenate((r_portf_n.T, r_portf_s.T, r_portf_a.T),axis=1)
        r_portf = pd.DataFrame(r_portf, index=x[int(0.5*len(x.index)):].index, columns=['1/N', 'Standard', 'Autoencoder'])
        
        r_excess.to_csv('./data/results/' + index + '_excess_returns.csv')
        r_portf.to_csv('./data/results/' + index + '_portfolio_returns.csv')
        
    return r_portf_n.T, r_excess_n.T, r_portf_s.T, r_excess_s.T, r_portf_a.T, r_excess_a.T

 
    
# get out-of-sample return vectors
index = 'CDAX'
returns = data.import_data(index + '_without_penny_stocks')
mktrf, rf = data.get_rf('daily')
x = data.join_risky_with_riskless(returns, rf)
     
r_portf_n, r_excess_n, r_portf_s, r_excess_s, r_portf_a, r_excess_a = run(x,index,1,True)







#def rolling_window(index, window_size):
#    # use rolling window only to reestimate portfolio weights, not to train autoencoder again
#    # Work in Progress
#    returns = data.import_data(index)
#    results = np.zeros((len(returns.index),3))
#    
#    for row in range(0, len(returns.index), 1):
#        x = data.x_data[row: row + window_size]
#        results[row] = run(3, x)
#        
#    return results


#def bootstrap_performance(data, n):
#    # bootstrapping sharpe ratio - Work in Progress
#    bs = StationaryBootstrap(252, data)
#    results = bs.apply(sharpe_ratio, 2500)
#    SR = pd.DataFrame(results[:,-1:], columns=['SR'])
#    fig = SR.hist(bins=40)
#    return



# expect to introduce a bias and reduce the varianc, use performance of portfolio as measure
# can do some statistical tests, confidence interval for sharpe ratio and compare to standard mean-var portfolio
# either reduce bias or variance
# estimates sensitive to use of rolling windows etc. (outliers entering window)
# denoising autoencoder is focused on reducing noise/variance of estimator
# mean-var portfolio is subject to erratic changes in weights - more transaction costs
# autoencoder reduces estimation variance, less extreme rebalancing, less turnover hopefully!
# compute turnover
# sharpe ratio relevant because investor wanted a certain sharpe ratio, tracking error
# using autoencoder to reduce turnover/variance
# CIs: bootstrapping samples or cross-validation to show significance, respecting sample properties, time series
# use seed for keras, test significance over time
# what are we estimating: mean&var or a portfolio??
# portfolio mean and variance not the same as the ones we are estimating (individual asset returns and covariances)
# in practice due to measurement error no direct link between the two
