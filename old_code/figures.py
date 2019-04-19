# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:59:56 2019

@author: plagl
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd

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
    #    # plot the series
    plt.figure(figsize=(10, 4), dpi=80)
    plt.plot(x.mean(axis=1), label=x_label)
    plt.plot(y.mean(axis=1), label=y_label)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend([x_label, y_label], loc='upper left')
    plt.xticks(np.arange(1, len(x.index), 252*2))
    plt.show()
    
#    z = x.mean(axis=1)
#    z = z.to_frame()
#    z.rename(columns={0:x_label}, inplace=True)
#    z[y_label] = y.mean(axis=1).values
##    z.index = pd.to_datetime(z.index)
##    z = z.rename_axis('date')
#    z.plot()
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

def plot_data(data, title, ylabel):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = data.mean(axis=1).plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.show()
    return
    
def plot_histogram(data):
    daily_returns = data.values.flatten()
    plt.hist(daily_returns, bins=5)
    mean = daily_returns.mean()
    std = daily_returns.std()
 
    plt.axvline(x=mean, color='r', linestyle='--')
    plt.axvline(x=std, color='k', linestyle='--')
    plt.axvline(x=-std, color='k', linestyle='--')
    plt.show()
    return


    
    
    