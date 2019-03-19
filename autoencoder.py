# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:46:47 2019

@author: plagl
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.models import load_model
from sklearn.preprocessing import StandardScaler  
from collections import defaultdict
#from google.colab import files
import io 
from scipy.optimize import minimize


def import_data(index):
    # Give the location of the file 
    script_path = os.getcwd()
    os.chdir( script_path )
    file = './data/' + index + '.csv'
    x=pd.read_csv(file, index_col=0).astype('float32')
    return x


def initialize_weights(num_stock):
    # initial guesses for the weights
    y0 = np.zeros(num_stock)
    for i in range(0,num_stock):
      y0[i]=1/num_stock
    y0=np.matrix(y0)
    return y0


def geometric_mean(x):    
    num_col=len(x.columns)
    num_rows=len(x.index)
    r_avg=np.ones((1,num_col))
    r_avg=np.matrix(r_avg)
    for j in range(0,num_col):
      ret=1
      for i in range(0,num_rows):
        ret= ret*(1+x.iloc[i,j])
      r_avg[0,j]=ret**(1/num_rows)
    r_avg=r_avg-np.ones((num_col)) 
    
    return r_avg
    
    
def one_over_N(x):
    num_obs=len(x.index)
    num_stock=len(x.columns)
    in_fraction=int(0.8*num_obs)
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    
    #construct portfolios    
    r_avg = geometric_mean(x_in)
    r_avg_oos = geometric_mean(x_oos)
    sigma_oos=np.cov(x_oos,rowvar=False)
    sigma_oos=np.matrix(sigma_oos)
    sigma=np.cov(x_in, rowvar=False)
    sigma=np.matrix(sigma)
      
    # weights such that they are all 1/num_stock
    y0 = np.zeros(num_stock)
    for i in range(0,num_stock):
      y0[i]=1/num_stock
    y0=np.matrix(y0)
    
    # in sample performance
    returns_in=((1+np.matmul(y0,np.transpose(r_avg)))**252-1)  
    # out of sample performance
    returns_oos=((1+np.matmul(y0,np.transpose(r_avg_oos)))**252-1)  
    print("returns in sample:", returns_in, "\nreturns out of sample:", returns_oos)
    
    
    

def mean_var_portfolio(x, y0):
    num_obs=len(x.index)
    in_fraction=int(0.8*num_obs)
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    num_stock=len(x.columns)
    min_ret=0.1 
    
    #construct mean-variance portfolios
    r_avg = geometric_mean(x_in)
    r_avg_oos = geometric_mean(x_oos)
 
    sigma_oos=np.cov(x_oos,rowvar=False)
    sigma_oos=np.matrix(sigma_oos)
    sigma=np.cov(x_in, rowvar=False)
    sigma=np.matrix(sigma)
    
    # maximize returns given a certain volatility
    def objective_standard(y):
      return np.sum(np.sqrt(252*np.matmul(np.matmul(y,sigma),np.transpose(y))))
    
    def constraint1(y):
      return np.sum(y)-1
    
    def constraint2(y):
      return np.sum(((1+np.matmul(y,np.transpose(r_avg)))**252-1))-(min_ret)
      
    # optimize
    b = [0,1] #bounds
    bnds=[np.transpose(b)] * num_stock   #vector of b's, with length num_stock
    con1 = {'type': 'eq', 'fun': constraint1} 
    con2= {'type': 'ineq', 'fun': constraint2}
    cons = ([con1, con2])
    solution = minimize(objective_standard,y0,method='SLSQP',\
                        bounds=bnds,constraints=cons)
    weights_standard=np.round(solution.x,4)
    
    # out of sample performance
    returns_standard=((1+np.matmul(weights_standard,np.transpose(r_avg_oos)))**252-1)        
    volatility_standard=np.sqrt(252*np.matmul(np.matmul(weights_standard,sigma_oos),np.transpose(weights_standard)))
    sharpe_standard=returns_standard/volatility_standard
    
    #print("returns standard:", returns_standard, "\nvolatility_standard:", volatility_standard, "\nsharpe_standard:", sharpe_standard)
    #print(weights_standard)
    return returns_standard, volatility_standard, sharpe_standard
   
    
    
def autoencoded_portfolio(x, y0):
    num_obs=len(x.index)
    num_stock=len(x.columns)
    in_fraction=int(0.8*num_obs)
    x_in=x[:in_fraction]
    x_oos=x[in_fraction:]
    min_ret=0.1
    
#    r_avg=x_in.mean()
#    r_avg=np.matrix(r_avg)
#    r_avg_oos=x_oos.mean()
#    r_avg_oos=np.matrix(r_avg_oos)
#    x_in_geo=1+x_in
#    r_avg=(np.cumprod(x_in_geo,axis=0).iloc[-1])**(1/in_fraction) - 1
#    x_oos_geo=1+x_oos
#    r_avg_oos=(np.cumprod(x_oos_geo,axis=0).iloc[-1])**(1/(num_obs-in_fraction)) - 1
    r_avg = geometric_mean(x_in)
    r_avg_oos = geometric_mean(x_oos)
    sigma_oos=np.cov(x_oos,rowvar=False)
    sigma_oos=np.matrix(sigma_oos)
    sigma=np.cov(x_in, rowvar=False)
    sigma=np.matrix(sigma)
    
    inp = Input(shape=(num_stock,))
    encoded = Dense(128, activation='relu')(inp)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(num_stock, activation='linear')(decoded)
    
    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy', 'mean_squared_error'])
    history = autoencoder.fit(x_in, x_in,
                    epochs=500,
                    batch_size=32,
                    shuffle=False,
                    validation_split=0.15, verbose=0)
    auto_data=pd.DataFrame(autoencoder.predict(x_in))
    #with pd.option_context('display.max_rows', 25, 'display.max_columns', None):
       # print(auto_data)
       
    # maximize returns given a certain volatility
    def objective_auto(y):
      return np.sum(np.sqrt(252*np.matmul(np.matmul(y,auto_sigma),np.transpose(y)))) 
  
    def constraint1(y):
      return np.sum(y)-1
  
    def constraint2(y):
      return np.sum(((1+np.matmul(y,np.transpose(r_avg)))**252-1))-(min_ret)
    
    def constraint2_auto(y):
      return np.sum(((1+np.matmul(y,np.transpose(auto_r_avg)))**252-1))-(min_ret)
  
    # optimize
    b = [0,1] #bounds
    bnds=[np.transpose(b)] * num_stock   #vector of b's, with length num_stock
    con1 = {'type': 'eq', 'fun': constraint1} 

    for i in range(0,num_stock):
      average=x_in.iloc[:,i].mean()
      average_auto=auto_data.iloc[:,i].mean()
      #auto_data.iloc[:,i]=auto_data.iloc[:,i]*average/average_auto
      stdev=x_in.iloc[:,i].std()
      stdev_auto=auto_data.iloc[:,i].std()
      auto_data.iloc[:,i]=stdev/stdev_auto*(np.transpose(np.matrix(auto_data.iloc[:,i]))-average_auto*np.ones((in_fraction,1)))+average*np.ones((in_fraction,1))
      
    auto_r_avg=auto_data.mean()
    auto_r_avg=np.matrix(auto_r_avg)
    auto_sigma=np.cov(auto_data, rowvar=False)
    auto_sigma=np.matrix(auto_sigma)
      
    con2_auto= {'type': 'ineq', 'fun': constraint2_auto}
    cons_auto = ([con1, con2_auto])
    solution_auto = minimize(objective_auto,y0,method='SLSQP',\
                        bounds=bnds,constraints=cons_auto)
    weights_auto=np.round(solution_auto.x,4)
    
    # out of sample performance
    returns_auto=((1+np.matmul(weights_auto,np.transpose(r_avg_oos)))**252-1)   
    volatility_auto=np.sqrt(252*np.matmul(np.matmul(weights_auto,sigma_oos),np.transpose(weights_auto))) 
    sharpe_auto=returns_auto/volatility_auto
    
    #print("returns auto:", returns_auto, "\nvolatility_auto:", volatility_auto, "\nsharpe_auto:", sharpe_auto, "\nweights_auto:", weights_auto)
    return returns_auto, volatility_auto, sharpe_auto, history


def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return


def plot_accuracy(history):  
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return
    
    
def run(num_trials, index):
    x = import_data(index)
    y0 = initialize_weights(len(x.columns))
    
    returns_s = np.zeros(num_trials)
    volatility_s = np.zeros(num_trials)
    sharpe_s = np.zeros(num_trials)
    returns_a = np.zeros(num_trials)
    volatility_a = np.zeros(num_trials)
    sharpe_a = np.zeros(num_trials)

    for n in range(0, num_trials):
        returns_standard, volatility_standard, sharpe_standard = mean_var_portfolio(x, y0)
        returns_auto, volatility_auto, sharpe_auto, history = autoencoded_portfolio(x, y0)
        plot_accuracy(history)
        plot_loss(history)
        returns_s[n], volatility_s[n], sharpe_s[n] = returns_standard, volatility_standard, sharpe_standard
        returns_a[n], volatility_a[n], sharpe_a[n] = returns_auto, volatility_auto, sharpe_auto
        
    avg_return_s = sum(returns_s) / num_trials
    avg_vol_s = sum(volatility_s) / num_trials
    avg_sharpe_s = sum(sharpe_s) / num_trials
    
    avg_return_a = sum(returns_a) / num_trials
    avg_vol_a = sum(volatility_a) / num_trials
    avg_sharpe_a = sum(sharpe_a) / num_trials
    
    print("returns standard:", avg_return_s, "\nvolatility_standard:", avg_vol_s, "\nsharpe_standard:", avg_sharpe_s)
    print("\nreturns auto:", avg_return_a, "\nvolatility_auto:", avg_vol_a, "\nsharpe_auto:", avg_sharpe_a)
    return returns_s, volatility_s, sharpe_s, returns_a, volatility_a, sharpe_a, history


       
returns_s, volatility_s, sharpe_s, returns_a, volatility_a, sharpe_a, history = run(1, 'CAC')



