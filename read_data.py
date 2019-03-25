# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:23:08 2019

@author: plagl
"""

# Reading an excel file using Python 
import pandas as pd
import os

def get_returns(index, remove_penny_stocks=False):
    # Give the location of the file 
    script_path = os.getcwd()
    os.chdir( script_path )
      
    # Assign spreadsheet filename to `file`
    file = './data/raw_data/' + index + '.xlsx'
    
    # Load spreadsheet into dataframe
    df = pd.read_excel(file, header=4, index_col=0)
    df = df.drop(df.columns[df.columns.str.contains('unnamed', case = False)], axis = 1)
    
    # Split datatypes into different data frames
    prices = df.filter(like='(P#T)~U$')
    sizes = df.filter(like='(MV)~U$')
    price_earnings = df.filter(like='(PE)')
    
    if remove_penny_stocks == True:
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
    #returns.to_csv('./data/' + index + '_without_penny_stocks.csv')
    return prices, sizes, price_earnings, returns                    



def compute_descriptives(returns):   
    mean_i = returns.mean(axis = 1)
    
    minimum = mean_i.min()
    maximum = mean_i.max()
    mean = mean_i.mean()
    variance = mean_i.var()
    skewness = mean_i.skew()
    kurtosis = mean_i.kurtosis()
    
    print(minimum)
    print(maximum)
    print(mean)
    print(variance)
    print(skewness)
    print(kurtosis)
   
prices, sizes, price_earnings, returns = get_returns('SSE', remove_penny_stocks=False)
compute_descriptives(returns)