# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:18:55 2019

@author: Fabrice
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
    sigma_in=np.cov(x_in,rowvar=False)
      
    # construct 1/N portfolio 
    y0 = np.matrix(np.ones(num_stock)*(1/num_stock))

    # in sample performance
    returns_in=((1 + y0*r_avg.T)**252) - 1
    
    # out of sample performance
    returns_oos=(1+ y0*r_avg_oos.T)**252 - 1     
    volatility_oos=np.sqrt(252 * y0*sigma_oos*y0.T)
    sharpe_oos=returns_oos/volatility_oos
    volatility_in = np.sqrt(252 * y0*sigma_in*y0.T)
    sharpe_in = returns_in/volatility_in
    
    print("returns in sample:", returns_in,"\nvolatility in sample:",volatility_in, "\nsharpe in sample:",sharpe_in, "\n\nreturns out of sample:", returns_oos, "\nvolatility out of sample:", volatility_oos, "\nsharpe out of sample:", sharpe_oos)
    return returns_in, returns_oos, volatility_oos, sharpe_oos
    

#x = import_data('FTSE_without_penny_stocks')
x = import_data('SSE')
#returns_in, returns_oos, volatility_oos, sharpe_oos = one_over_N(x)
    #encoded_data, auto_data = autoencode_data(x, epochs=50, batch_size=64, activations='relu', depth=3, neurons=100)
      
returns_in, volatility_oos, sharpe_oos, returns_oos = one_over_N(x)

