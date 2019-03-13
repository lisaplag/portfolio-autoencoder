# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:23:08 2019

@author: plagl
"""

# Reading an excel file using Python 
import pandas as pd
import os

def get_returns():
    # Give the location of the file 
    script_path = os.getcwd()
    os.chdir( script_path )
      
    # Assign spreadsheet filename to `file`
    file = './data/raw data/SES.xlsx'
    
    # Load spreadsheet into dataframe
    df = pd.read_excel(file, header=4, index_col=0)
    df = df.drop(df.columns[df.columns.str.contains('unnamed', case = False)], axis = 1)
    
    # Split datatypes into different data frames
    prices = df.filter(like='(P#T)~U$')
    sizes = df.filter(like='(MV)~U$')
    price_earnings = df.filter(like='(PE)')
    
    # Compute returns and reduce sample to limit NA values
    returns = prices.pct_change()
    returns = returns.loc['2003-12-31':'2018-12-31', :]
    returns = returns.dropna(axis = 1)
    
    returns = returns.loc[(returns!=0).any(axis=1)]
    returns = returns.loc[:, (returns != float('Inf')).all(axis=0)]

    
    # Write returns to CSV file
    returns.to_csv('./data/SES.csv')
    return sizes, price_earnings, returns                    

sizes, price_earnings, returns = get_returns()


def compute_descriptives(returns):
    min_t = returns.min(axis = 1)
    minimum = min_t.min()
    max_t = returns.max(axis = 1)
    maximum = max_t.max()
    
    mean_i = returns.mean(axis = 0)
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
    
compute_descriptives(returns)