# -*- coding: utf-8 -*-
"""
@author: tobiashoogteijling
"""

import numpy as np
import read_data as data
import tensorflow as tf
import random as rn
from keras import backend as K
import pandas as pd
import math
from keras.layers.advanced_activations import LeakyReLU, ReLU, ELU
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import HDF5Matrix

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

good = pd.DataFrame(outcomes_good[1:], columns=['Chi2', 'Pesaran', 'Portmanteau1', 'Portmanteau3', 'Portmanteau5','MSPE_r', 'MSPE_sigma'])
good.to_csv('./data/results/outcomes_good.csv')

data1 = data.import_data('./results/outcomes_good')
data2=data.import_data('./results/MSPE_5_outcomes')

dat=np.concatenate((data1,data2[1:]),axis=0)
st_mean=np.sqrt(np.var(dat[:,5]))
m_mean=dat[:,5].mean()
t_mean=(m_mean-MSPE_r)/st_mean*np.sqrt(dat.shape[0])

st_var=np.sqrt(np.var(dat[:,6]))
m_var=dat[:,6].mean()
t_var=(m_var-MSPE_sigma)/st_var*np.sqrt(dat.shape[0])

