# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:23:08 2019

@author: plagl
"""

import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def get_returns(index, remove_penny_stocks=False, write=False):
    # Give the location of the file 
    script_path = os.getcwd()
    os.chdir( script_path )
      
    # Assign spreadsheet filename to `file`
    file = './data/raw_data/' + index + '.xlsx'
    
    # Load spreadsheet into dataframe
    df = pd.read_excel(file, header=4, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.drop(df.columns[df.columns.str.contains('unnamed', case = False)], axis = 1)
    
    # Split datatypes into different data frames
    prices = df.filter(like='(P#T)~U$')
    sizes = df.filter(like='(MV)~U$')
    price_earnings = df.filter(like='(PE)')
    
    if remove_penny_stocks:
        prices = prices.loc[:, (prices >= 1).all(axis=0)]
        
    # Compute returns and reduce sample to limit NA values
    returns = prices.pct_change()
    returns = returns.loc['2003-12-31':'2018-12-31', :]
    returns = returns.dropna(axis = 1)
    
    # Remove non-trading days
    returns = returns.loc[(returns!=0).any(axis=1)]
    # Remove infinite returns
    returns = returns.loc[:, (returns != float('Inf')).all(axis=0)]

    # Write returns to CSV file
    if write and remove_penny_stocks:
        returns.to_csv('./data/' + index + '_without_penny_stocks.csv')
    elif write:
        returns.to_csv('./data/' + index + '.csv')
        
    return prices, sizes, price_earnings, returns      


def get_rf(frequency='daily', descriptives=False):
    # Give the location of the file 
    script_path = os.getcwd()
    os.chdir( script_path )
      
    # Assign spreadsheet filename to `file`
    file = './data/RF_' + frequency + '.csv'
    
    # Load spreadsheet into dataframe
    df = pd.read_csv(file, header=0, index_col=0)
    df.index = pd.to_datetime(df.index)
    rf = df.filter(items=['rf'])
    mktrf = df.filter(items=['mktrf'])
    
    
    # Compute descriptive statistics if desired
    if descriptives:
        print(rf.min())
        print(rf.max())
        print(rf.mean())
        print(rf.var())
        print(rf.skew())
        print(rf.kurtosis())
        print(stats.jarque_bera(rf))
    
    return mktrf, rf


def join_risky_with_riskless(risky, riskless):
    risky = risky.assign(key=1)
    riskless = riskless.assign(key=1)
    merged = pd.merge(risky, riskless, on='key').drop('key',axis=1)
              
    return merged


def compute_descriptives(returns):   
    mean_i = returns.mean(axis = 1)
    
    minimum = mean_i.min()
    maximum = mean_i.max()
    mean = mean_i.mean()
    variance = mean_i.var()
    skewness = mean_i.skew()
    kurtosis = mean_i.kurtosis()
    jb = stats.jarque_bera(mean_i)
    
    print(minimum)
    print(maximum)
    print(mean)
    print(variance)
    print(skewness)
    print(kurtosis)
    print(jb)
    
def plot_data(data, title, ylabel):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = data.mean(axis=1).plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.show()
    
def plot_histogram(data):
    daily_returns = data.values.flatten()
    plt.hist(daily_returns, bins=5)
    mean = daily_returns.mean()
    std = daily_returns.std()
 
    plt.axvline(x=mean, color='r', linestyle='--')
    plt.axvline(x=std, color='k', linestyle='--')
    plt.axvline(x=-std, color='k', linestyle='--')
 
    plt.show()
    
   
#prices, sizes, price_earnings, returns = get_returns('NASDAQ', remove_penny_stocks=False, write=False)
#compute_descriptives(returns)

#plot_histogram(returns)
    
mktrf, rf = get_rf('daily', True)


