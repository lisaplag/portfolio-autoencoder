# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:46:47 2019

@author: tobiashoogteijling
"""

import numpy as np
import read_data as data
import tensorflow as tf
import random as rn
from keras import backend as K
import pandas as pd
import math
from keras.layers.advanced_activations import LeakyReLU, ReLU, ELU
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import HDF5Matrix
from scipy.optimize import minimize
from read_data import get_rf, join_risky_with_riskless
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
import math

def chi2test(u):
    num_pos = 0
    T = np.size(u, 0)
    N = np.size(u, 1)
    u = np.matrix(u)
    num_pos = sum(n > 0 for n in u).sum()
    chi2 = 4 * np.square(num_pos - 0.5 * N * T) / N / T
    return chi2


def pesarantest(u):
    T = np.size(u, 0)
    N = np.size(u, 1)
    CD = 0
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            CD = CD + np.corrcoef(u.iloc[:, i], u.iloc[:, j])[0, 1]
    CD = np.sqrt(2 * T / N / (N - 1)) * CD
    return CD


def portmanteau(u, h):
    T = np.size(u, 0)
    N = np.size(u, 1)
    C = np.zeros((h + 1, N, N))
    Q = 0
    for k in range(0, h + 1):
        for i in range(1 + k, T):
            C[k, :, :] = np.add(C[k, :, :], np.outer(u.iloc[i, :], u.iloc[i - k, :]))
        C[k, :, :] = C[k, :, :] / T
    C0_inv = np.linalg.inv(C[0, :, :])
    for k in range(1, h + 1):
        Q = Q + 1 / (T - k) * np.trace(np.transpose(C[h, :, :]) * C0_inv * C[h, :, :] * C0_inv)
    Q = Q * T * T
    return Q


def advanced_autoencoder(x_in, x, epochs, batch_size, activations, depth, neurons):
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    num_stock = len(x_in.columns)

    # activation functions
    if activations == 'elu':
        function = ELU(alpha=1.0)
    elif activations == 'lrelu':
        function = LeakyReLU(alpha=0.1)
    else:
        function = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)

    autoencoder = Sequential()
    # encoding layers of desired depth
    for n in range(1, depth + 1):
        # input layer
        if n == 1:
            # autoencoder.add(GaussianNoise(stddev=0.01, input_shape=(num_stock,)))
            autoencoder.add(Dense(int(neurons / n), input_shape=(num_stock,)))
            autoencoder.add(function)
        else:
            autoencoder.add(Dense(int(neurons / n)))
            autoencoder.add(function)
    # decoding layers of desired depth
    for n in range(depth, 1, -1):
        autoencoder.add(Dense(int(neurons / (n - 1))))
        autoencoder.add(function)
    # output layer
    autoencoder.add(Dense(num_stock, activation='linear'))

    # autoencoder.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['accuracy'])

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.txt', verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None,
                                 restore_best_weights=True)
    history = autoencoder.fit(x_in, x_in, epochs=epochs, batch_size=batch_size, \
                              shuffle=False, validation_split=0.15, verbose=0, callbacks=[earlystopper])
    # errors = np.add(autoencoder.predict(x_in),-x_in)
    y = autoencoder.predict(x)
    # saving results of error distribution tests
    # A=np.zeros((5))
    # A[0]=chi2test(errors)
    # A[1]=pesarantest(errors)
    # A[2]=portmanteau(errors,1)
    # A[3]=portmanteau(errors,3)
    # A[4]=portmanteau(errors,5)

    # autoencoder.summary()

    # plot accuracy and loss of autoencoder
    # plot_accuracy(history)
    # plot_loss(history)

    # plot original, encoded and decoded data for some stock
    # plot_two_series(x_in, 'Original data', auto_data, 'Reconstructed data')

    # the histogram of the data
    # make_histogram(x_in, 'Original data', auto_data, 'Reconstructed data')

    # CLOSE TF SESSION
    K.clear_session()
    return y

def MVO(mu, Sigma, min_ret):
    mu = np.array(mu)
    Sigma = np.array(Sigma)
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
    cons = [{'type': 'eq', 'fun': weight_constraint},
            {'type': 'ineq', 'fun': return_constraint}]

    # Minimize
    solution = minimize(objective_function, w0, method='SLSQP', bounds=bnds, constraints=cons)
    weights = solution.x
    return weights.reshape((N,1))


def adaptive_threshold_EWMA(e, tau, t):
    e = np.array(e)
    T,N = e.shape
    ecov_roll = np.array(pd.DataFrame(e).ewm(alpha=0.03).cov())
    ecov_roll = np.nan_to_num(ecov_roll)
    ecov = np.zeros((T,N,N))
    for t in range(T):
        ecov[t,:,:] = ecov_roll[t*N:t*N+N]
    adapted_ecov = np.array(ecov.copy())
    theta = np.zeros((T,N,N))
    for t in range(T):
        print(t)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                else:
                    theta[t,i,j] = 0.03 * np.square(e[t,i] * e[t,j] - ecov[t,i,j]) + 0.97 * theta[t-1,i,j]

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                elif np.abs(ecov[t,i,j]) < np.sqrt(theta[t,i,j]) * tau:
                    adapted_ecov[t,i,j] = 0

    n_nonzeros = np.count_nonzero(adapted_ecov)-N
    fraction_restored = n_nonzeros/(T*N*(N-1))
    return adapted_ecov, fraction_restored

dataset = data.import_data('CDAX_without_penny_stocks')
mktrf, rf = get_rf('daily', True)
dataset = join_risky_with_riskless(dataset, rf)
rf_merged = np.array(dataset['rf'])
rf_merged = rf_merged.reshape((rf_merged.shape[0],1))

np.random.seed(1)
rn.seed(12345)
tf.set_random_seed(1234)

num_obs = dataset.shape[0]

in_fraction = int(0.5 * num_obs)
first_period = num_obs
x_in = dataset.iloc[:in_fraction, :]
num_stock = dataset.shape[1]  # not including the risk free stock
chi2_bound = 6.635
z_bound = 2.58
runs = 1
labda = 0.97
s = 500
x = np.matrix(dataset.iloc[:first_period, :])
num_obs = first_period
# predictions standard
r_pred = np.zeros((num_obs, num_stock))
s_pred = np.zeros((num_obs, num_stock, num_stock))
s_pred[0, :num_stock, :num_stock] = np.outer((r_pred[0:1, :num_stock]), (r_pred[0:1, :num_stock]))
weights = np.zeros((num_obs - in_fraction, num_stock))
portfolio_ret = np.zeros((num_obs - in_fraction, 1))
portfolio_vol = np.zeros((num_obs - in_fraction, 1))
MSPE_sigma = 0

for i in range(1, num_obs):
    if i < s + 1:
        r_pred[i:i + 1, :num_stock] = x[0:i, :num_stock].mean(axis=0)
    else:
        r_pred[i:i + 1, :num_stock] = x[i - s:i, :num_stock].mean(axis=0)
    s_pred[i, :num_stock, :num_stock] = (1 - labda) * np.outer((x[i - 1:i, :num_stock] - r_pred[i - 1:i, :num_stock]), (
                x[i - 1:i, :num_stock] - r_pred[i - 1:i, :num_stock])) + labda * s_pred[i - 1, :num_stock, :num_stock]

f_errors = r_pred - x
MSPE_r = np.square(f_errors[num_obs - in_fraction:, :num_stock]).mean()
for i in range(num_obs - in_fraction, num_obs):
    MSPE_sigma = MSPE_sigma + np.square(
        np.outer(f_errors[i:i + 1, :], f_errors[i:i + 1, :]) - s_pred[i, :num_stock, :num_stock]).mean()
MSPE_sigma = MSPE_sigma / (num_obs - in_fraction)

np.random.seed(121)
rn.seed(1212345)
tf.set_random_seed(121234)
# prediction autoencoded data

MSPE_sigma_auto_diag = np.array(np.zeros((num_obs,1)))
MSPE_sigma_auto_threshold = np.array(np.zeros((num_obs,1)))
t = in_fraction
out_fraction = num_obs-in_fraction
x_in_norf = x_in.iloc[:, :-1]
x_norf = np.matrix(np.array(x)[:, :-1])
finished = False
portfolio_returns_diag = np.zeros((num_obs,1))
portfolio_returns_threshold = np.zeros((num_obs,1))
while finished is False:
    print('t = ', t)
    test_passed = False
    counter = 0
    while test_passed == False:
        counter += 1
        auto_data = advanced_autoencoder(x_in_norf, x_norf, 1000, 10, 'elu', 3, 100)
        auto_data = np.matrix(auto_data)
        errors = pd.DataFrame(np.add(auto_data[:in_fraction, :], -x_in_norf))
        A = np.zeros((5))
        A[0] = chi2test(errors)
        A[1] = pesarantest(errors)
        A[2] = portmanteau(errors, 1)
        A[3] = portmanteau(errors, 3)
        A[4] = portmanteau(errors, 5)
        if (A[0] < chi2_bound and abs(A[1]) < z_bound):
            auto_data = np.append(auto_data, np.array(x[:, -1]), axis=1)
            num_stock = auto_data.shape[1]
            r_pred_auto = np.zeros((num_obs, num_stock))
            s_pred_auto = np.zeros((num_obs, num_stock, num_stock))
            s_pred_auto[0, :num_stock, :num_stock] = np.outer((r_pred_auto[0:1, :num_stock]),
                                                              (r_pred_auto[0:1, :num_stock]))

            weights_auto = np.zeros((num_obs - in_fraction, num_stock))
            portfolio_ret_auto = np.zeros((num_obs - in_fraction, 1))
            portfolio_vol_auto = np.zeros((num_obs - in_fraction, 1))
            resids = pd.DataFrame(auto_data-x)


            for i in range(1, num_obs):
                if i < s + 1:
                    r_pred_auto[i, :num_stock] = auto_data[0:i, :num_stock].mean(axis=0)
                else:
                    r_pred_auto[i, :num_stock] = auto_data[i - s:i, :num_stock].mean(axis=0)
                s_pred_auto[i, :num_stock, :num_stock] = (1 - labda) * np.outer(
                    (auto_data[i - 1, :num_stock] - r_pred_auto[i - 1, :num_stock]),
                    (auto_data[i - 1, :num_stock] - r_pred_auto[i - 1, :num_stock])) + labda * s_pred_auto[i - 1,:num_stock, :num_stock]

            f_errors_auto = r_pred_auto - x
            MSPE_r_auto = np.square(f_errors_auto[t:t+252, :num_stock]).mean()

            # Add residual volatility
            resids_vol = resids.ewm(alpha=0.03).var()
            s_pred_auto_diag = s_pred_auto.copy()
            for i in range(1, num_obs):
                for j in range(0, num_stock-1):
                    s_pred_auto_diag[i, j, j] = s_pred_auto_diag[i, j, j] + resids_vol.iloc[i,j]

            for i in range(t, t+252):
                MSPE_sigma_auto_diag[i] = np.square(np.outer(f_errors_auto[i:i + 1, :], f_errors_auto[i:i + 1, :]) -
                                               s_pred_auto_diag[i, :num_stock, :num_stock]).mean()

            auto_weights_diag = MVO(r_pred_auto[t,:], s_pred_auto_diag[t,:,:], 0.0001)
            log_returns_diag = np.log(x[t:t+252, :]+1)
            portfolio_returns_diag[t:t+252] = log_returns_diag @ auto_weights_diag

            # Threshold
            adapted_ecov, fraction_restored = adaptive_threshold_EWMA(resids, 0.25, t)
            s_pred_auto_threshold = s_pred_auto + adapted_ecov

            for i in range(t, t+252):
                MSPE_sigma_auto_threshold[i] = np.square(np.outer(f_errors_auto[i, :], f_errors_auto[i, :]) -
                                               s_pred_auto_threshold[i, :num_stock, :num_stock]).mean()

            auto_weights_threshold = MVO(r_pred_auto[t,:], s_pred_auto_threshold[t,:,:], 0.0001)
            portfolio_returns_threshold[t:t+252] = x[t:t+252, :] @ auto_weights_threshold
            test_passed = True

    if t == num_obs - 252:
        finished = True
    if num_obs - t < 504:
        t = num_obs - 252
    else:
        t = t + 252

pd.DataFrame(np.concatenate([portfolio_returns_threshold, portfolio_returns_diag], axis=1)).to_csv('yearly_portfolio_returns.csv')
pd.DataFrame(np.concatenate([MSPE_sigma_auto_threshold, MSPE_sigma_auto_diag], axis = 1)).to_csv('yearly_MSPE.csv')