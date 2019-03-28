# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:59:56 2019

@author: plagl
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
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
    
def plot_two_series(x, x_label, y, y_label):
    # plot the series
    plt.figure(figsize=(10, 4), dpi=80)
    plt.plot(x.mean(axis=1), label=x_label)
    plt.plot(y.mean(axis=1), label=y_label)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend([x_label, y_label], loc='upper left')
    plt.xticks(np.arange(1, len(x.index), 252*2))
    plt.show()
    return
    
def make_histogram(x, x_label, y, y_label):
    # the histogram of the data
    range_hist = (-0.1, 0.1)
    n, bins, patches = plt.hist([x.values.flatten(), y.values.flatten()], \
                                 bins=20, range = range_hist)
    # add a 'best fit' line
#    y = mlab.normpdf(bins, 0, 1)
#    plt.plot(20, y, 'r--')
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.legend([x_label, y_label])
    plt.show()
    return