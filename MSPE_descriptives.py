# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:32:06 2019

@author: plagl
"""
import numpy as np
import pandas as pd
import read_data as data


def analyze_MSPE(index):
    dataset = data.import_data(index)
    num_obs=dataset.shape[0]
    
    in_fraction=int(0.5*num_obs)
    first_period=num_obs
    x_in=dataset.iloc[:in_fraction,:]
    num_stock=dataset.shape[1] #not including the risk free stock
    chi2_bound=6.635
    z_bound=2.58
    runs=1
    labda=0.94
    s=500
    x=np.matrix(dataset.iloc[:first_period,:])
    num_obs=first_period
    # predictions standard
    r_pred=np.zeros((num_obs,num_stock))
    s_pred=np.zeros((num_obs,num_stock,num_stock))
    s_pred[0,:num_stock,:num_stock]=np.outer((r_pred[0:1,:num_stock]),(r_pred[0:1,:num_stock]))
    weights=np.zeros((num_obs-in_fraction,num_stock))
    portfolio_ret=np.zeros((num_obs-in_fraction,1))
    portfolio_vol=np.zeros((num_obs-in_fraction,1))
    MSPE_sigma=0
               
    for i in range(1,num_obs):
      if i<s+1:
        r_pred[i:i+1,:num_stock]=x[0:i,:num_stock].mean(axis=0)
      else:
        r_pred[i:i+1,:num_stock]=x[i-s:i,:num_stock].mean(axis=0)
      s_pred[i,:num_stock,:num_stock]=(1-labda)*np.outer((x[i-1:i,:num_stock]-r_pred[i-1:i,:num_stock]),(x[i-1:i,:num_stock]-r_pred[i-1:i,:num_stock]))+labda*s_pred[i-1,:num_stock,:num_stock]
    
    f_errors=r_pred-x
    MSPE_r=np.square(f_errors[num_obs-in_fraction:,:num_stock]).mean()
    for i in range(num_obs-in_fraction,num_obs):
      MSPE_sigma=MSPE_sigma+np.square(np.outer(f_errors[i:i+1,:],f_errors[i:i+1,:])-s_pred[i,:num_stock,:num_stock]).mean()
    MSPE_sigma=MSPE_sigma/(num_obs-in_fraction)
    
    
    MSPE_data = data.import_data('results/' + 'MSPE_5_outcomes')
    mean = MSPE_data['MSPE_r']
    sigma = MSPE_data['MSPE_sigma']
    
    min_r = mean.min()
    max_r = mean.max()
    mean_r = mean.mean()
    var_r = mean.var()
    
    print(min_r)
    print(max_r)
    print(mean_r)
    print(var_r)
    print('\n')
    
    
    min_s = sigma.min()
    max_s = sigma.max()
    mean_s = sigma.mean()
    var_s = sigma.var()
    
    print(min_s)
    print(max_s)
    print(mean_s)
    print(var_s)
    
    return MSPE_r, MSPE_sigma
    



index = 'CDAX_without_penny_stocks'
r, sigma = analyze_MSPE(index)


