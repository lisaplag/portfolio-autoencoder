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

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

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
        
    #make noisy data  
    num_obs_in=len(x_in.index)
    in_fraction=int(0.85*num_obs_in)
    x_train=x_in[:in_fraction] #in-smaple training subsample
    x_test=x_in[in_fraction:] #in-sample testing subsample
    
    noise_factor = 0.01
    x_train_noisy = x_train + noise_factor * np.random.standard_t(df=3, size=x_train.shape)
         
        
    autoencoder = Sequential()
    # encoding layers of desired depth
    for n in range(1, depth+1):
        # input layer
        if n==1:           
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
    
    #checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.txt', verbose=0, save_best_only=True)
    earlystopper=EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0,mode='auto',baseline=None,restore_best_weights=True)
    history=autoencoder.fit(x_train_noisy, x_train, epochs=epochs, batch_size=batch_size, \
                              shuffle=False, validation_data=(x_test,x_test), verbose=0,callbacks=[earlystopper])
    errors = np.add(autoencoder.predict(x_in),-x_in)
#    test123 = autoencoder.predict(x_in)
#    test1231 = test123.var()
#    print(test1231)
    # saving results of error distribution tests
    A=np.zeros((5))
    A[0]=chi2test(errors)
    A[1]=pesarantest(errors)
    A[2]=portmanteau(errors,1)
    A[3]=portmanteau(errors,3)
    A[4]=portmanteau(errors,5)
        
    #autoencoder.summary()
    
    # plot accuracy and loss of autoencoder
    # plot_accuracy(history)
    # plot_loss(history)
    
    # plot original, encoded and decoded data for some stock
    # plot_two_series(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    # the histogram of the data
    # make_histogram(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    #CLOSE TF SESSION
    K.clear_session()
    return A
#   return test123
    

dataset = data.import_data('CDAX_without_penny_stocks')
np.random.seed(1)
rn.seed(12345)        
tf.set_random_seed(1234)

num_obs=dataset.shape[0]



in_fraction=int(0.5*num_obs)
x_in=dataset.iloc[:in_fraction]
num_stock=dataset.shape[1] #not including the risk free stock

runs=1
labda=0.94
s=500

different_depths=[1,2,3,4,5]
different_neurons=[120,100,80]
results=np.zeros((5,3,5,100))

#resultstest=advanced_autoencoder(x_in,1000,10,'elu',3,100)
#print(resultstest)

# loop over different autoencoders
counter=0
for i in range(0,5):
  for j in range(0,3):
    np.random.seed(1)
    rn.seed(12345)        
    tf.set_random_seed(1234)
    for k in range(0,100):
      results[i,j,:,k]=advanced_autoencoder(x_in,1000,10,'elu',different_depths[i],different_neurons[j])
      print(results[i,j,:,k])
      counter=counter+1
      print(counter)
      
coun=0   
test_stats=np.zeros((1500,7))
for i in range(0,5):
    for j in range(0,3):
        for k in range(0,100):
            test_stats[coun,0]=different_depths[i]
            test_stats[coun,1]=different_neurons[j]
            test_stats[coun,2:]=results[i,j,:,k]
            coun=coun+1
            print(coun)
            
significant_stats=np.zeros((1,7))
chi2_bound=6.635
z_bound=2.58
portmanteau_mean1=np.square(num_stock)
portmanteau_mean2=np.square(num_stock)*3
portmanteau_mean3=np.square(num_stock)*5
portmanteau_stdev1=np.sqrt(2*portmanteau_mean1)
portmanteau_stdev2=np.sqrt(2*portmanteau_mean2)
portmanteau_stdev3=np.sqrt(2*portmanteau_mean3)
chi_counter=0
pesaran_counter=0
portmanteau1_counter=0
portmanteau3_counter=0
portmanteau5_counter=0
portmanteau_stats=np.zeros((1,7))
for i in range(0,1500):
    if abs((test_stats[i,4]-portmanteau_mean1)/portmanteau_stdev1)<z_bound:
        portmanteau_stats=np.concatenate((portmanteau_stats,np.matrix(test_stats[i,:])),axis=0)
        portmanteau1_counter=portmanteau1_counter+1 
    elif abs((test_stats[i,5]-portmanteau_mean2)/portmanteau_stdev2)<z_bound:
        portmanteau_stats=np.concatenate((portmanteau_stats,np.matrix(test_stats[i,:])),axis=0)   
        portmanteau3_counter=portmanteau3_counter+1 
    elif abs((test_stats[i,6]-portmanteau_mean3)/portmanteau_stdev3)<z_bound:
        portmanteau_stats=np.concatenate((portmanteau_stats,np.matrix(test_stats[i,:])),axis=0)   
        portmanteau5_counter=portmanteau5_counter+1 
for i in range(0,1500):
    if test_stats[i,2]<chi2_bound:
        significant_stats=np.concatenate((significant_stats,np.matrix(test_stats[i,:])),axis=0)
        chi_counter=chi_counter+1    
    elif abs(test_stats[i,3])<z_bound:
        significant_stats=np.concatenate((significant_stats,np.matrix(test_stats[i,:])),axis=0)
    elif abs((test_stats[i,4]-portmanteau_mean1)/portmanteau_stdev1)<z_bound:
        significant_stats=np.concatenate((significant_stats,np.matrix(test_stats[i,:])),axis=0)  
    elif abs((test_stats[i,5]-portmanteau_mean2)/portmanteau_stdev2)<z_bound:
        significant_stats=np.concatenate((significant_stats,np.matrix(test_stats[i,:])),axis=0)
    elif abs((test_stats[i,6]-portmanteau_mean3)/portmanteau_stdev3)<z_bound:
        significant_stats=np.concatenate((significant_stats,np.matrix(test_stats[i,:])),axis=0)
    if abs(test_stats[i,3])<z_bound:
        pesaran_counter=pesaran_counter+1    
#    if abs((test_stats[i,4]-portmanteau_mean1)/portmanteau_stdev1)<z_bound:
#        portmanteau1_counter=portmanteau1_counter+1    
#    if abs((test_stats[i,5]-portmanteau_mean2)/portmanteau_stdev2)<z_bound:
#        portmanteau3_counter=portmanteau3_counter+1
#    if abs((test_stats[i,6]-portmanteau_mean3)/portmanteau_stdev3)<z_bound:
#        portmanteau5_counter=portmanteau5_counter+1

significant_stats2=np.zeros((1,7))
for i in range(0,1500):
    if test_stats[i,2]<chi2_bound:
        significant_stats2=np.concatenate((significant_stats2,np.matrix(test_stats[i,:])),axis=0)   
    elif abs(test_stats[i,3])<z_bound:
        significant_stats2=np.concatenate((significant_stats2,np.matrix(test_stats[i,:])),axis=0)

significant_stats3=np.zeros((1,7))
for i in range(0,1500):
    if test_stats[i,2]<chi2_bound and abs(test_stats[i,3])<z_bound:
        significant_stats3=np.concatenate((significant_stats3,np.matrix(test_stats[i,:])),axis=0)   
