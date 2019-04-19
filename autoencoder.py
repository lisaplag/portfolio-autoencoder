# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:46:47 2019

@author: plagl
"""

import numpy as np
import read_data as data
import tensorflow as tf
import random as rn
import math
from keras import backend as K
import pandas as pd
from keras.layers.advanced_activations import LeakyReLU, ReLU, ELU
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import HDF5Matrix
import scipy

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)



# old version
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
    return auto_data





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
    
    # saving results of error distribution tests
    errors = np.add(autoencoder.predict(x_in),-x_in)
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

    
    

