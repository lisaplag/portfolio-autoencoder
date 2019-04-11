# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:26:42 2019

@author: plagl
"""

import numpy as np
import read_data as data
import tensorflow as tf
import random as rn
from keras import backend as K
import pandas as pd
from keras.layers.advanced_activations import LeakyReLU, ReLU, ELU
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
    
    # train the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    earlystopper=EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0,mode='auto',baseline=None,restore_best_weights=True)
    history=autoencoder.fit(x_in, x_in, epochs=epochs, batch_size=batch_size, \
                              shuffle=False, validation_split=0.15, verbose=0,callbacks=[earlystopper])
    errors = np.add(autoencoder.predict(x_in),-x_in)
    
    # saving results of error distribution tests
    A=np.zeros((5))
    A[0]=chi2test(errors)
    A[1]=pesarantest(errors)
    A[2]=portmanteau(errors,1)
    A[3]=portmanteau(errors,3)
    A[4]=portmanteau(errors,5)
        
    #autoencoder.summary()
    
    #CLOSE TF SESSION
    K.clear_session()
    return A


def get_stats(index, iterations, depths, neurons, write=True):
    dataset = data.import_data(index)
    np.random.seed(511)
    rn.seed(5123451)        
    tf.set_random_seed(512341)
    
    num_obs=dataset.shape[0]
    num_stock=dataset.shape[1] #not including the risk free stock
    in_fraction=int(0.5*num_obs)
    x_in=dataset.iloc[:in_fraction]
    
    # loop over different autoencoders
    counter=0
    stats=np.zeros((15*iterations, 7))
    for i in range(0,5):
      for j in range(0,3):
        np.random.seed(511)
        rn.seed(5123451)        
        tf.set_random_seed(512341)
        for k in range(0, iterations):
          stats[counter,0]= different_depths[i]
          stats[counter,1]= different_neurons[j]
          stats[counter,2:]=advanced_autoencoder(x_in,1000,10,'elu',different_depths[i],different_neurons[j])
          counter=counter+1
          print(counter)
                
    sig_stats=np.zeros((1,7))
    chi2_bound=6.635
    z_bound=2.58
    
    portmanteau_mean1=np.square(num_stock)
    portmanteau_mean2=np.square(num_stock)*3
    portmanteau_mean3=np.square(num_stock)*5
    portmanteau_stdev1=np.sqrt(2*portmanteau_mean1)
    portmanteau_stdev2=np.sqrt(2*portmanteau_mean2)
    portmanteau_stdev3=np.sqrt(2*portmanteau_mean3)
    
    chi_count=0
    pesaran_count=0
    portmanteau1_count=0
    portmanteau3_count=0
    portmanteau5_count=0
    
    for i in range(0, 15*iterations):
        if stats[i,2]<chi2_bound:
            sig_stats=np.concatenate((sig_stats,np.matrix(stats[i,:])),axis=0)
            chi_count=chi_count+1    
        elif abs(stats[i,3])<z_bound:
            sig_stats=np.concatenate((sig_stats,np.matrix(stats[i,:])),axis=0)
        elif abs((stats[i,4]-portmanteau_mean1)/portmanteau_stdev1)<z_bound:
            sig_stats=np.concatenate((sig_stats,np.matrix(stats[i,:])),axis=0)  
        elif abs((stats[i,5]-portmanteau_mean2)/portmanteau_stdev2)<z_bound:
            sig_stats=np.concatenate((sig_stats,np.matrix(stats[i,:])),axis=0)
        elif abs((stats[i,6]-portmanteau_mean3)/portmanteau_stdev3)<z_bound:
            sig_stats=np.concatenate((sig_stats,np.matrix(stats[i,:])),axis=0)
        if abs(stats[i,3])<z_bound:
            pesaran_count=pesaran_count+1    
        if abs((stats[i,4]-portmanteau_mean1)/portmanteau_stdev1)<z_bound:
            portmanteau1_count=portmanteau1_count+1    
        if abs((stats[i,5]-portmanteau_mean2)/portmanteau_stdev2)<z_bound:
            portmanteau3_count=portmanteau3_count+1
        if abs((stats[i,6]-portmanteau_mean3)/portmanteau_stdev3)<z_bound:
            portmanteau5_count=portmanteau5_count+1
    
    sig_stats2=np.zeros((1,7))
    for i in range(0, 15*iterations):
        if stats[i,2]<chi2_bound:
            sig_stats2=np.concatenate((sig_stats2,np.matrix(stats[i,:])),axis=0)   
        elif abs(stats[i,3])<z_bound:
            sig_stats2=np.concatenate((sig_stats2,np.matrix(stats[i,:])),axis=0)
    
    sig_stats3=np.zeros((1,7))
    for i in range(0, 15*iterations):
        if stats[i,2]<chi2_bound and abs(stats[i,3])<z_bound:
            sig_stats3=np.concatenate((sig_stats3,np.matrix(stats[i,:])),axis=0)   
        
    if write:
        # removing 0 row and preparing for writing to csv
        stats1 = pd.DataFrame(sig_stats[1:], columns=['Depth', 'Neurons', 'Chi2', 'Pesaran', 'Portmanteau1', 'Portmanteau3', 'Portmanteau5'])
        stats1.to_csv('./data/results/' + index + '_stats1.csv')
        
        stats2 = pd.DataFrame(sig_stats2[1:], columns=['Depth', 'Neurons', 'Chi2', 'Pesaran', 'Portmanteau1', 'Portmanteau3', 'Portmanteau5'])
        stats2.to_csv('./data/results/' + index + '_stats2.csv')
        
        stats3 = pd.DataFrame(sig_stats3[1:], columns=['Depth', 'Neurons', 'Chi2', 'Pesaran', 'Portmanteau1', 'Portmanteau3', 'Portmanteau5'])
        stats3.to_csv('./data/results/' + index + '_stats3.csv')
        
        c_list = [chi_count, pesaran_count, portmanteau1_count, portmanteau3_count, portmanteau5_count]
        counts = pd.DataFrame(c_list, index=['Chi2', 'Pesaran', 'Portmanteau1', 'Portmanteau3', 'Portmanteau5'], columns=['counts'])
        counts.to_csv('./data/results/' + index + '_counts.csv')
        
    return counts, stats1, stats2, stats3



def aggregate_stats(stats, depths, neurons):
    counts=np.zeros((len(neurons),len(depths)))
    n=0
    d=0
    # counting significant results per depth d and number of neurons n
    for i in range(0,len(stats['Depth'])):
        print(stats['Depth'][i])
        d = depths.index(stats['Depth'][i])
        n = neurons.index(stats['Neurons'][i])
        counts[n,d] += 1
     
    counts_df = pd.DataFrame(counts, index=neurons, columns=depths)
    return counts_df
    
    
    

index = 'CDAX_without_penny_stocks'
iterations=100
different_depths=[1,2,3,4,5]
different_neurons=[120,100,80]

#counts, stat1, stat2, stat3 = get_stats(index, iterations, different_depths, different_neurons, True)

stat1 = data.import_data('results/' + index + '_stats1')
stat2 = data.import_data('results/' + index + '_stats2')
stat3 = data.import_data('results/' + index + '_stats3')

sig_counts1 = aggregate_stats(stat1, different_depths, different_neurons)
sig_counts2 = aggregate_stats(stat2, different_depths, different_neurons)
sig_counts3 = aggregate_stats(stat3, different_depths, different_neurons)






