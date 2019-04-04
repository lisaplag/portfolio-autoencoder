# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:35:20 2019

@author: plagl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cvxopt as opt
from cvxopt import blas, solvers

np.random.seed(123)

# Turn off progress printing 
#solvers.options['show_progress'] = False


def import_data(index):
    # Give the location of the file 
    script_path = os.getcwd()
    os.chdir( script_path )
    file = './data/' + index + '.csv'
    x=pd.read_csv(file, index_col=0)
    return x
    

def random_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''
    p = np.asmatrix(np.mean(returns, axis=0))
    w = np.asmatrix(random_weights(returns.shape[1]))
    C = np.asmatrix(np.cov(returns, rowvar=False))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
#    if sigma > 0.01:
#        return random_portfolio(returns)
    return mu, sigma



def generate_portfolios(returns, n_portfolios):
    means, stds = np.column_stack([random_portfolio(returns) for _ in range(n_portfolios)])
    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')
    return means, stds


def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.asmatrix(returns)
    p = np.asmatrix(np.mean(returns, axis=0))

    mus = [0.0001,0.0002]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns,rowvar=False))
    pbar = opt.matrix(p)
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


returns = import_data('CAC')
means, stds = generate_portfolios(returns, 50)
weights, returns, risks = optimal_portfolio(returns)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')
    


