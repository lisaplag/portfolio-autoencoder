# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:46:47 2019

@author: tobiashoogteijling
"""

import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(1)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)


import pandas as pd
import math
from keras.layers.advanced_activations import LeakyReLU, ReLU, ELU
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, Sequential

def advanced_autoencoder(x_in, epochs, batch_size, activations, depth, neurons):
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    num_stock=len(x_in.columns)
    
    # activation functions    
    if activations == 'elu':
        function = ELU(alpha=1.0)
    elif activations == 'lrelu':
        function = LeakyReLU(alpha=0.1)
    else:
        function = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
        
    autoencoder = Sequential()
    # encoding layers of desired depth
    for n in range(1, depth+1):
        # input layer
        if n==1:
            #autoencoder.add(GaussianNoise(stddev=0.01, input_shape=(num_stock,)))
            autoencoder.add(Dense(int(neurons/n), input_shape=(num_stock,)))
            autoencoder.add(function)
        else:            
            autoencoder.add(Dense(int(neurons/n)))
            autoencoder.add(function)
    # decoding layers of desired depth
    for n in range(depth, 1, -1):
        autoencoder.add(Dense(int(neurons/(n-1))))
        autoencoder.add(function)
    # output layer
    autoencoder.add(Dense(num_stock, activation='linear'))
    
    #autoencoder.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['accuracy'])
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    e=0
    A=np.zeros((5,20))
    for i in range(0,20):
        history=autoencoder.fit(x_in, x_in, epochs=epochs, batch_size=batch_size, \
                              shuffle=False, validation_split=0.15, verbose=0)
        errors = np.add(autoencoder.predict(x_in),-x_in)
        A[0,i]=chi2test(errors)
        A[1,i]=pesarantest(errors)
        A[2,i]=portmanteau(errors,1)
        A[3,i]=portmanteau(errors,3)
        A[4,i]=portmanteau(errors,5)
        
    #autoencoder.summary()
    
    # plot accuracy and loss of autoencoder
   # plot_accuracy(history)
   # plot_loss(history)
    
    # plot original, encoded and decoded data for some stock
    #plot_two_series(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    # the histogram of the data
    #make_histogram(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    #print(x_in.mean(axis=0).mean())
    #print(x_in.std(axis=0).mean())
    #print(auto_data.mean(axis=0).mean())
    #print(auto_data.std(axis=0).mean())

    #CLOSE TF SESSION
    K.clear_session()
    #with pd.option_context('display.max_rows', 25, 'display.max_columns', None):
    #print(auto_data)
    return A
    
def chi2test(u):
  num_pos=0
  T=np.size(u,0)
  N=np.size(u,1)
  u=np.matrix(u)
  num_pos=sum(n>0 for n in u).sum()
  chi2=4*np.square(num_pos-0.5*N*T)/N/T
  return chi2

def pesarantest(u):
  T=np.size(u,0)
  N=np.size(u,1)
  CD=0
  for i in range(0,N-1):
    for j in range(i+1,N):
      CD=CD+np.corrcoef(u.iloc[:,i],u.iloc[:,j])[0,1]
  CD=np.sqrt(2*T/N/(N-1))*CD  
  return CD  

def portmanteau(u,h):
  T=np.size(u,0)
  N=np.size(u,1)
  C=np.zeros((h+1,N,N))
  Q=0
  for k in range(0,h+1):
    for i in range(1+k,T):
      C[k,:,:]=np.add(C[k,:,:],np.outer(u.iloc[i,:],u.iloc[i-k,:]))
    C[k,:,:]=C[k,:,:]/T
  C0_inv=np.linalg.inv(C[0,:,:])
  for k in range(1,h+1):
    Q=Q+1/(T-k)*np.trace(np.transpose(C[h,:,:])*C0_inv*C[h,:,:]*C0_inv)
  Q=Q*T*T
  return Q

#from keras import regularizers
#from keras.models import load_model
#from sklearn.preprocessing import StandardScaler  
#from collections import defaultdict
#from sklearn.decomposition import PCA
#from arch.bootstrap import StationaryBootstrap
# -*- coding: u
dataset=pd.read_csv('NASDAQ_without_penny_stocks.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
data=dataset.iloc[:,:]
num_obs=np.size(data,0)
in_fraction=int(0.5*num_obs)

x_in=dataset.iloc[:in_fraction]
num_stock=np.size(data,1) #not including the risk free stock
runs=1
labda=0.94
s=500

different_depths=[1,2,3,4,5]
different_neurons=[800,700,600,500,400]
results=np.zeros((5,5,5,20))




for i in range(0,5):
  for j in range(0,5):
    np.random.seed(1)
    rn.seed(12345)        
    tf.set_random_seed(1234)
    results[i,j,:,:]=advanced_autoencoder(x_in,100,10,'elu',different_depths[i],different_neurons[j])
    print(results[i,j,:,:])
      


