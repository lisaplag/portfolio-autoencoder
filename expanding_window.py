# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:46:47 2019

@author: plagl
"""

import numpy as np
import os
import pandas as pd
import math
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, Sequential
from scipy.optimize import minimize
import figures as fg
import read_data as read

from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU, ELU


def import_data(index, risk_free_asset=True):
    # location of the files
    script_path = os.getcwd()
    os.chdir( script_path )
    file = './data/' + index + '.csv'
    x = pd.read_csv(file, index_col=0)
    return x

def expanding_window(index, window_size, startup_period=504, min_ret=0.001):
    # Work in Progress
    data = import_data(index)
    T, N = data.shape
    data = np.array(data)
    leftover = (T - startup_period) % window_size
    startup_period += leftover
    M = int((T - startup_period) / window_size)
    R_o_portfolio = np.empty(0)
    R_a_portfolio = np.empty(0)

    # Measurements
    MSFE_o = 0
    MSFE_a = 0

    # Construct a portfolio for each month
    for m in range(M):
        t = startup_period + m * window_size
        R_m = data[t:t + window_size]
        mu_m, Sigma_m = np.mean(R_m, 0), np.cov(np.transpose(R_m))

        R_o = data[:t]
        mu_o, Sigma_o = np.mean(R_m, 0), np.cov(np.transpose(R_o))
        MSFE_o += np.mean((np.triu(Sigma_o - Sigma_m, 1)) ** 2)
        w_o = MVO(mu_o, Sigma_o, min_ret)
        R_o_m = R_m @ w_o
        R_o_portfolio = np.append(R_o_portfolio, R_o_m)

        R_a = autoencoder_window(R_o)
        mu_a, Sigma_a = np.mean(R_a, 0), np.cov(np.transpose(R_a))
        Omega = adaptive_threshold(R_a, R_o-R_a, 0.1)
        Sigma_a = Sigma_a + Omega
        MSFE_a += np.mean((np.triu(Sigma_a - Sigma_m, 1)) ** 2)
        w_a = MVO(mu_o, Sigma_a, min_ret)
        R_a_m = R_m @ w_a
        R_a_portfolio = np.append(R_a_portfolio, R_a_m)

    return R_a_portfolio, R_o_portfolio

def autoencoder_window(x_in):
    num_obs, N = x_in.shape

    inputs = Input(shape=(N,))
    h1 = Dense(64)(inputs)
    z1 = LeakyReLU(0.1)(h1)
    outputs = Dense(N, activation='linear')(z1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mae')
    model.fit(x_in, x_in, batch_size=32, epochs=70, verbose=0)
    x_fitted = model.predict(x_in)
    return x_fitted

def MVO(mu, Sigma, min_ret):
    N = mu.shape[0]

    # Define optimization problem
    objective_function = lambda w : np.transpose(w) @ Sigma @ w
    weight_constraint = lambda w : np.sum(w) - 1
    return_constraint = lambda w : np.transpose(w) @ mu - min_ret

    # Initialize
    w0 = np.zeros((1, N))
    w0[:] = 1/N
    b = [0, 1] # bounds
    bnds = [np.transpose(b)] * N
    cons = ({'type': 'eq', 'fun': weight_constraint},
            {'type': 'ineq', 'fun': return_constraint})

    # Minimize
    solution = minimize(objective_function, w0, method='SLSQP', bounds=bnds, constraints=cons)
    weights = solution.x
    return weights

def adaptive_threshold(R, e, tau):
    Rcov = np.cov(np.transpose(R))
    ecov = np.cov(np.transpose(e))
    N = Rcov.shape[0]
    adapted_ecov = ecov.copy()
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            else:
                e1e2 = e[:,i] * e[:,j]
                theta = e1e2.var()
                if e1e2.mean() < np.sqrt(theta)*tau:
                    adapted_ecov[i,j] = 0
    return adapted_ecov

x = import_data('NASDAQ_without_penny_stocks')
index = 'NASDAQ_without_penny_stocks'
window_size = 21
startup_period = 504
min_ret = 0.001

portfolio_returns_a, portfolio_returns_o = expanding_window(index,window_size,startup_period,min_ret)