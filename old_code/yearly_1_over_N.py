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
    Sigma = np.array(Sigma)
    N = mu.shape[0]
    mu = np.array(mu).reshape((N,1))


    # Define optimization problem
    objective_function = lambda w : np.transpose(w) @ Sigma @ w
    weight_constraint = lambda w : np.sum(w) - 1
    return_constraint = lambda w : np.transpose(w) @ mu - min_ret

    # Initialize
    w0 = np.zeros((N,1))
    w0[:] = 0
    w0[-1] = 1
    b = [0, 0.05] # bounds
    bnds = [np.transpose(b)] * (N-1)
    bnds.append(np.transpose([0, 0.5]))
    cons = [{'type': 'eq', 'fun': weight_constraint},
            {'type': 'ineq', 'fun': return_constraint}]

    # Minimize
    solution = minimize(objective_function, w0, method='SLSQP', bounds=bnds, constraints=cons, options={'ftol': 1e-30})
    weights = solution.x
    return weights.reshape((N,1))



dataset = data.import_data('CDAX_without_penny_stocks')
mktrf, rf = get_rf('daily', False)
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
labda = 0.94
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
portfolio_returns_original = np.zeros((num_obs,1))
portfolio_returns_oneoverN = np.zeros((num_obs,1))
weights_original = np.zeros((num_obs,num_stock))
weights_oneoverN = np.zeros((num_obs,num_stock))
MSPE_oneoverN = np.zeros((num_obs,1))
MSPE_originalMVO = np.zeros((num_obs,1))

while finished is False:
    print('t = ', t)
    counter = 0

    # Original returns MVO
    originalMVO_weights = MVO(r_pred[t, :], s_pred[t, :, :], 0.0001)
    originalMVO_weights_norf = originalMVO_weights / (1 - originalMVO_weights[-1])
    originalMVO_weights_norf[-1] = 0
    for i in range(t, t + 252):
        portfolio_returns_original[i] = x[i, :] @ originalMVO_weights_norf
        originalMVO_weights_norf = originalMVO_weights_norf * np.array(1 + x[i, :]).transpose() / sum(
            originalMVO_weights_norf)
        weights_original[i, :] = originalMVO_weights_norf.transpose()

    # 1/N
    oneoverN_weights = np.zeros((num_stock,1)) + 1/num_stock
    oneoverN_weights_norf = oneoverN_weights / (1 - oneoverN_weights[-1])
    oneoverN_weights_norf[-1] = 0
    for i in range(t, t + 252):
        portfolio_returns_oneoverN[i] = x[i, :] @ oneoverN_weights_norf
        oneoverN_weights_norf = oneoverN_weights_norf * np.array(1 + x[i, :]).transpose() / sum(
            oneoverN_weights_norf)
        weights_oneoverN[i, :] = oneoverN_weights_norf.transpose()



    if t == num_obs - 252:
        finished = True
    if num_obs - t < 504:
        t = num_obs - 252
    else:
        t = t + 252

log_returns_original = np.log(portfolio_returns_original+1)
log_returns_oneoverN = np.log(portfolio_returns_oneoverN+1)

pd.DataFrame(np.concatenate([log_returns_original, log_returns_oneoverN], axis=1)).to_csv('./data/results/Yearly_portfolio/yearly_portfolio_returns_oneoverN_maxweights_CDAX.csv')
