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

from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU, ELU
from keras import regularizers
from keras.models import load_model
from sklearn.preprocessing import StandardScaler  
from collections import defaultdict
from sklearn.decomposition import PCA




def import_data(index):
    # location of the files 
    script_path = os.getcwd()
    os.chdir( script_path )
    file = './data/' + index + '.csv'
    x=pd.read_csv(file, index_col=0)
    return x


def initialize_weights(num_stock):
    # initial guesses for the weights
    y0 = np.matrix(np.ones(num_stock)*(1/num_stock))
    return y0


    
    
def one_over_N(x):
    num_obs=len(x.index)
    num_stock=len(x.columns)
    in_fraction=int(0.8*num_obs)
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    
    # compute means and covariance matrix
    r_avg=np.asmatrix(np.mean(x_in, axis=0))
    r_avg_oos=np.asmatrix(np.mean(x_oos, axis=0))
    sigma_oos=np.cov(x_oos,rowvar=False)
      
    # construct 1/N portfolio 
    y0 = np.matrix(np.ones(num_stock)*(1/num_stock))

    # in sample performance
    returns_in=((1 + y0*r_avg.T)**252) - 1
    
    # out of sample performance
    returns_oos=(1+ y0*r_avg_oos.T)**252 - 1     
    volatility_oos=np.sqrt(252 * y0*sigma_oos*y0.T)
    sharpe_oos=returns_oos/volatility_oos
    
    print("returns in sample:", returns_in, "\nreturns out of sample:", returns_oos)
    return returns_in, returns_oos, volatility_oos, sharpe_oos
    

def mean_var_portfolio(x, y0):
    num_obs=len(x.index)
    in_fraction=int(0.8*num_obs)
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    num_stock=len(x.columns)
    min_ret=0.1 
    
    #construct mean-variance portfolios
    r_avg=np.asmatrix(np.mean(x_in, axis=0))
    r_avg_oos=np.asmatrix(np.mean(x_oos, axis=0))
    sigma_oos=np.cov(x_oos,rowvar=False)
    sigma=np.cov(x_in, rowvar=False)
    
    # maximize returns given a certain volatility
    def objective_standard(y):
        y=np.asmatrix(y)
        return np.sum(np.sqrt(252 * y*sigma*y.T))
    def constraint1(y):
        y=np.asmatrix(y)
        return np.sum(y) - 1
    def constraint2(y):
        y=np.asmatrix(y)
        return np.sum((1 + y*r_avg.T)**252 - 1) - (min_ret)
      
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
    returns_standard=(1 + weights_standard*r_avg_oos.T)**252 - 1      
    volatility_standard=np.sqrt(252 * weights_standard*sigma_oos*weights_standard.T)
    sharpe_standard=returns_standard/volatility_standard
    
    #print("returns standard:", returns_standard, "\nvolatility_standard:", volatility_standard, "\nsharpe_standard:", sharpe_standard)
    #print(weights_standard)
    #print(sum(weights_standard))
    return returns_standard, volatility_standard, sharpe_standard
   


def autoencode_data(x_in, epochs, batch_size, activations, depth, neurons):
    num_stock=len(x_in.columns)
    inp = Input(shape=(num_stock,))
    
    # activation functions
    def gelu(x):
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
    def relu(x):
        return max(x, 0)
    def lrelu(x):
        return max(0.01*x, x)
    
#    if activations == 'gelu':
#        function = gelu(x)
#    elif activations == 'lrelu':
#        function = lrelu(x)
#    else:
#        function = relu(x)
    
    # encoding layers of desired depth
    for n in range(1, depth+1):
        if n == 1:
            # input layer
            encoded = Dense(int(neurons/n), activation=activations)(inp)
        else:
            encoded = Dense(int(neurons/n), activation=activations)(encoded)
    # decoding layers of desired depth
    for n in range(depth, 1, -1):
        if n == depth:
            # bottleneck
            decoded = Dense(int(neurons/(n-1)), activation=activations)(encoded)
        else:   
            decoded = Dense(int(neurons/(n-1)), activation=activations)(decoded)
    # output layer
    decoded = Dense(num_stock, activation='linear')(decoded)
    
    autoencoder = Model(inp, decoded)
    encoder = Model(inp, encoded)
    autoencoder.summary()
    #autoencoder.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['accuracy'])
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = autoencoder.fit(x_in, x_in, epochs=epochs, batch_size=batch_size, \
                              shuffle=False, validation_split=0.15, verbose=0)
    encoded_data=pd.DataFrame(encoder.predict(x_in))
    auto_data=pd.DataFrame(autoencoder.predict(x_in))
    
    fg.plot_accuracy(history)
    fg.plot_loss(history)
    
    # plot original, encoded and decoded data for some stock
    fg.plot_two_series(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    # the histogram of the data
    fg.make_histogram(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    print(x_in.mean(axis=0).mean())
    print(x_in.std(axis=0).mean())
    print(auto_data.mean(axis=0).mean())
    print(auto_data.std(axis=0).mean())

    #with pd.option_context('display.max_rows', 25, 'display.max_columns', None):
    #print(auto_data)
    return auto_data




def advanced_autoencoder(x_in, epochs, batch_size, activations, depth, neurons):
    num_stock=len(x_in.columns)
    inp = Input(shape=(num_stock,))
    
    # activation functions
    def gelu(x):
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
    def relu(x):
        return max(x, 0)
    def lrelu(x):
        return max(0.01*x, x)
    
    #make noisy data  
    num_obs=len(x.index)
    in_fraction=int(0.85*num_obs)
    x_train=x_in[:in_fraction]
    x_test=x_in[in_fraction:]
    
    noise_factor = 0.05
    x_train_noisy = x_in + noise_factor * np.random.standard_t(df=3, size=x_in.shape)
     
    # encoding layers of desired depth
    for n in range(1, depth+1):
        if n == 1:
            # input layer
            encoded = Dense(int(neurons/n), activation=activations)(inp)
        else:
            encoded = Dense(int(neurons/n), activation=activations)(encoded)
    # decoding layers of desired depth
    for n in range(depth, 1, -1):
        if n == depth:
            # bottleneck
            decoded = Dense(int(neurons/(n-1)), activation=activations)(encoded)
        else:   
            decoded = Dense(int(neurons/(n-1)), activation=activations)(decoded)
    # output layer
    decoded = Dense(num_stock, activation='linear')(decoded)
    
    autoencoder = Model(inp, decoded)
    encoder = Model(inp, encoded)
    autoencoder.summary()
    #autoencoder.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['accuracy'])
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = autoencoder.fit(x_train_noisy, x_train, epochs=epochs, batch_size=batch_size, \
                              shuffle=False, validation_data=(x_test,x_test), verbose=0)
    encoded_data=pd.DataFrame(encoder.predict(x_in))
    auto_data=pd.DataFrame(autoencoder.predict(x_in))
    
#    fg.plot_accuracy(history)
#    fg.plot_loss(history)
    
    # plot original, encoded and decoded data for some stock
    fg.plot_two_series(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    # the histogram of the data
    fg.make_histogram(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    print(x_in.mean(axis=0).mean())
    print(x_in.std(axis=0).mean())
    print(auto_data.mean(axis=0).mean())
    print(auto_data.std(axis=0).mean())

    #with pd.option_context('display.max_rows', 25, 'display.max_columns', None):
    #print(auto_data)
    return auto_data



    
    
def autoencoded_portfolio(x, initial_weights, activation, depth, method):
    num_obs=len(x.index)
    num_stock=len(x.columns)
    in_fraction=int(0.8*num_obs)
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    min_ret=0.1
    
    r_avg_oos=np.asmatrix(np.mean(x_oos, axis=0))
    sigma_in=np.cov(x_in,rowvar=False)
    sigma_oos=np.cov(x_oos,rowvar=False)
    
    # autoencoding in-sample data
    auto_data = advanced_autoencoder(x_in=x_in, epochs=50, batch_size=32, \
                                     activations=activation, depth=depth, neurons=int(num_stock/2))

    # rescaling autoencoded data to original mean and variance
    if method == 'rescale':
        for i in range(0,num_stock):
          average=x_in.iloc[:,i].mean()
          average_auto=auto_data.iloc[:,i].mean()
          stdev=x_in.iloc[:,i].std()
          stdev_auto=auto_data.iloc[:,i].std()
          auto_data.iloc[:,i]=stdev/stdev_auto * ((np.matrix(auto_data.iloc[:,i])).T \
                        - average_auto*np.ones((in_fraction,1))) + average*np.ones((in_fraction,1))
        auto_r_avg=np.mean(auto_data, axis=0)
        auto_sigma=np.cov(auto_data, rowvar=False)
    # set diagonal elements of autoencoded data covariance matrix equal to original variance
    elif method == 'original_variance':
        auto_r_avg=np.mean(auto_data, axis=0)
        auto_sigma=np.cov(auto_data, rowvar=False)
        for i in range(0,num_stock):
            auto_sigma[i,i]=sigma_in[i,i]
    # no rescaling at all
    else:
        auto_r_avg=np.mean(auto_data, axis=0)
        auto_sigma=np.cov(auto_data, rowvar=False)
        
    auto_r_avg=np.asmatrix(auto_r_avg)
    
    # minimize volatility given target return
    def objective_auto(y):
        y=np.asmatrix(y)
        return np.sum(np.sqrt(252 * y*auto_sigma*y.T)) 
    def constraint1(y):
        y=np.asmatrix(y)
        return np.sum(y)-1
    def constraint2_auto(y):
        y=np.asmatrix(y)
        return np.sum((1+ y*auto_r_avg.T )**252 - 1) - (min_ret)
  
    # optimize
    b = [0,1] #bounds
    bnds=[np.transpose(b)] * num_stock   #vector of b's, with length num_stock
    con1 = {'type': 'eq', 'fun': constraint1} 
    con2_auto= {'type': 'ineq', 'fun': constraint2_auto}
    cons_auto = ([con1, con2_auto])
    solution_auto = minimize(objective_auto,initial_weights,method='SLSQP',\
                             bounds=bnds,constraints=cons_auto)
    weights_auto=np.asmatrix(solution_auto.x)
    
    # evaluate out of sample performance
    returns_auto=(1 + weights_auto*r_avg_oos.T)**252 - 1 
    volatility_auto=np.sqrt(252 * weights_auto*sigma_oos*weights_auto.T) 
    sharpe_auto=returns_auto/volatility_auto
    
    #print(weights_auto)
    #print(sum(weights_auto))
    #print("returns auto:", returns_auto, "\nvolatility_auto:", volatility_auto, "\nsharpe_auto:", sharpe_auto, "\nweights_auto:", weights_auto)
    return returns_auto, volatility_auto, sharpe_auto, auto_data
    
      
def run(num_trials, x):
    y0 = initialize_weights(len(x.columns))
    
    # construct 1/N portfolio
    return_in, return_oos, volatility_oos, sharpe_oos = one_over_N(x)
    
    # construct standard mean-variance portfolio
    return_s, volatility_s, sharpe_s = mean_var_portfolio(x, y0)
    
    # construct portfolios based on autoencoded returns 
    returns_a = np.zeros(num_trials)
    volatility_a = np.zeros(num_trials)
    sharpe_a = np.zeros(num_trials)
    for n in range(0, num_trials):
        np.random.seed(1) # using random seed to get reproducible results
        returns_auto, volatility_auto, sharpe_auto, auto_data = autoencoded_portfolio(x, y0, 'relu', 2, 'original_variance')
        returns_a[n], volatility_a[n], sharpe_a[n] = returns_auto, volatility_auto, sharpe_auto
        
    # average over the trials
    avg_return_a = sum(returns_a) / num_trials
    avg_vol_a = sum(volatility_a) / num_trials
    avg_sharpe_a = sum(sharpe_a) / num_trials
    
    print("\nreturns 1/N:", return_oos, "\nvolatility 1/N:", volatility_oos, "\nsharpe 1/N:", sharpe_oos)
    print("\nreturns standard:", return_s, "\nvolatility standard:", volatility_s, "\nsharpe standard:", sharpe_s)
    print("\nreturns auto:", avg_return_a, "\nvolatility auto:", avg_vol_a, "\nsharpe auto:", avg_sharpe_a)
    
    return return_s, volatility_s, sharpe_s, returns_a, volatility_a, sharpe_a, auto_data


def rolling_window(index, window_size):
    data = import_data(index)
    results = np.zeros((len(data.index),3))
    
    for row in range(0, len(data.index), 1):
        x = data.x_data[row: row + window_size]
        results[row] = run(3, x)
        
    return results

x = import_data('CDAX_without_penny_stocks')
#returns_in, returns_oos, volatility_oos, sharpe_oos = one_over_N(x)
    #encoded_data, auto_data = autoencode_data(x, epochs=50, batch_size=64, activations='relu', depth=3, neurons=100)
      
returns_s, volatility_s, sharpe_s, returns_a, volatility_a, sharpe_a, auto_data = run(50, x)
