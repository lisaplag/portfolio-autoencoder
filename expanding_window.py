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
    # use rolling window only to reestimate portfolio weights, not to train autoencoder again
    # Work in Progress
    data = import_data(index)
    T, N = data.shape
    leftover = (T - startup_period) % window_size
    startup_period = + leftover
    M = int((T - startup_period) / window_size)
    R_o_portfolio = np.empty(0)
    R_a_portfolio = np.empty(0)

    # Measurements
    MSFE_o = 0
    MSFE_a = 0
    for m in range(M):
        t = startup_period + m * window_size
        R_m = data.iloc[t:t + window_size]
        mu_m, Sigma_m = R_m.mean(0), R_m.cov()

        R_o = data.iloc[:t]
        mu_o, Sigma_o = R_o.mean(0), R_o.cov()
        MSFE_o = + np.mean((np.triu(Sigma_o - Sigma_m)) ** 2)
        w_o = MVO(mu_o, Sigma_o, min_ret)
        R_o_m = R_m @ w_o
        R_o_portfolio = np.append(R_o_portfolio, R_o_m)

        R_a = autoencoder_window(R_o)
        mu_a, Sigma_a = R_a.mean(0), R_a.cov()
        MSFE_a = + np.mean((np.triu(Sigma_a - Sigma_m)) ** 2)
        w_a = MVO(mu_a, Sigma_a, min_ret)
        R_a_m = R_m @ w_a
        R_a_portfolio = np.append(R_a_portfolio, R_a_m)


    print('day: ', t)
    print('Cumulative returns of original portfolio:', np.sum(R_o_portfolio))
    print('Cumulative returns of autoencoded portfolio', np.sum(R_a_portfolio))
    print('MSFE_o: ', MSFE_o)
    print('MSFE_a: ', MSFE_a)


def autoencoder_window(x_in):
    num_obs, N = x_in.shape

    inputs = Input(shape=(N,))
    h1 = Dense(64)(inputs)
    z1 = LeakyReLU(0.1)(h1)
    outputs = Dense(N, activation='linear')(z1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mae')
    model.fit(x_in, x_in, batch_size=32, epochs=70, verbose=0)
    x_fitted = pd.DataFrame(model.predict(x_in))
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

x = import_data('NASDAQ_without_penny_stocks')
index = 'NASDAQ_without_penny_stocks'
window_size = 21
startup_period = 504
min_ret = 0.001
