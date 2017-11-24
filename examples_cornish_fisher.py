# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:11:28 2017

@author: Jesus
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from cvar_optimization import standardize, rescale, approx_quantile
from print_context import print_context

# import real data
data = pd.read_excel('test_data.xlsx',sheetname='data_sample')
data = data.iloc[:,[1,2]]/100
data = data.values

# print its mean and variance-covariance
with print_context(formatter={'float': '{: 1.4f}'.format}):
    X = data
    print('mean:')
    print(X.mean(axis=0))
    print('covariance:')
    print(np.cov(X.T))
    print('skewness:')
    print(skew(X))
    print('(excess) kurtosis:')
    print(kurtosis(X))

# standardize and print its mean and variance-covariance (sanity check)
data_m_v = standardize(data)
with print_context(formatter={'float': '{: 1.4f}'.format}):
    X = data_m_v
    print('mean:')
    print(X.mean(axis=0))
    print('covariance:')
    print(np.cov(X.T))
    print('skewness:')
    print(skew(X))
    print('(excess) kurtosis:')
    print(kurtosis(X))

# new estimates of mean and variance-covariance matrices
new_mean = np.array([0.01, 0.04])
corr = np.array([[1.0, 0.8], [0.8, 1.0]])
new_std_dev = np.array([0.03, 0.07])
new_cov = np.outer(new_std_dev, new_std_dev)*corr
new_data = rescale(data_m_v, new_mean, new_cov)

# print its mean and variance-covariance
with print_context(formatter={'float': '{: 1.4f}'.format}):
    X = new_data
    print('mean:')
    print(X.mean(axis=0))
    print('covariance:')
    print(np.cov(X.T))
    print('skewness:')
    print(skew(X))
    print('(excess) kurtosis:')
    print(kurtosis(X))
    
# use Cornish-Fisher formulas to compute Value-at-Risk (VaR)
X = new_data[:,0]
Y = new_data[:,1]
quantile_x = -approx_quantile(X, q=0.05)
quantile_y = -approx_quantile(Y, q=0.05)
print('VaR for X: {:1.4f}'.format(quantile_x))
print('VaR for Y: {:1.4f}'.format(quantile_y))
    
# Compute Expected Shortfall or Conditional VaR using Cornish-Fisher approx.
cvar_x = 0
cvar_y = 0
for i in range(5):
    q = (i + 1)/100    
    cvar_x += approx_quantile(X, q=q)
    cvar_y += approx_quantile(Y, q=q)
cvar_x = -cvar_x/5
cvar_y = -cvar_y/5
    
print('CVaR for X: {:1.4f}'.format(cvar_x))
print('CVaR for Y: {:1.4f}'.format(cvar_y))