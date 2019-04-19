# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:35:04 2019

@author: plagl
"""

import numpy as np
import pandas as pd
import read_data as data


index = 'FTSE_without_penny_stocks_different_neurons'
chi2_bound=6.635
z_bound=2.58
stats3 = data.import_data('results/' + index + '_stats3')

sig_stats3=np.zeros((1,7))

for i in range(0, len(stats3.index)):
    if stats3['Chi2'][i]<chi2_bound and abs(stats3['Pesaran'][i])<z_bound:
        sig_stats3=np.concatenate((sig_stats3,np.matrix(stats3.iloc[i,:])),axis=0)
        
        
stats3 = pd.DataFrame(sig_stats3[1:], columns=['Depth', 'Neurons', 'Chi2', 'Pesaran', 'Portmanteau1', 'Portmanteau3', 'Portmanteau5'])
stats3.to_csv('./data/results/' + index + '_stats3.csv')