# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:46:47 2019

@author: plagl
"""

import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(3)
rn.seed(3)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(3)


import pandas as pd
import math
import figures as fg
from keras.layers.advanced_activations import LeakyReLU, ReLU, ELU
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, Sequential

#from keras import regularizers
#from keras.models import load_model
#from sklearn.preprocessing import StandardScaler  
#from collections import defaultdict
#from sklearn.decomposition import PCA
#from arch.bootstrap import StationaryBootstrap


# old version - not in use anymore
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
    history = autoencoder.fit(x_in, x_in, epochs=epochs, batch_size=batch_size, \
                              shuffle=False, validation_split=0.15, verbose=0)
    auto_data=pd.DataFrame(autoencoder.predict(x_in))
    autoencoder.summary()
    
    # plot accuracy and loss of autoencoder
    fg.plot_accuracy(history)
    fg.plot_loss(history)
    
    # plot original, encoded and decoded data for some stock
    #fg.plot_two_series(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    # the histogram of the data
    fg.make_histogram(x_in, 'Original data', auto_data, 'Reconstructed data')
    
    print(x_in.mean(axis=0).mean())
    print(x_in.std(axis=0).mean())
    print(auto_data.mean(axis=0).mean())
    print(auto_data.std(axis=0).mean())

    #CLOSE TF SESSION
    K.clear_session()

    return auto_data
    
    

