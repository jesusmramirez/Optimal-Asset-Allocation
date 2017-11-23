# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:39:33 2017

@author: Jesus
"""

import pandas as pd
import numpy as np
import cvar_optimization as opt
from print_context import print_context

# make fake data
np.random.seed(0)
fake_mean = np.array([0.08, 0.02])
fake_variance = np.array([[0.0025, 0.0006],[0.0006, 0.0009]])
data = np.random.multivariate_normal(mean=fake_mean,cov=fake_variance,size=1000)

# import real data
data = pd.read_excel('test_data.xlsx',sheetname='data_sample')
data = data.iloc[:,[1,2]]/100

# print its mean and variance-covariance
with print_context(formatter={'float': '{: 1.4f}'.format}):
    X = data
    print('mean:')
    print(X.mean(axis=0))
    print('covariance:')
    print(np.cov(X.T))

# standardize and print its mean and variance-covariance (sanity check)
data_m_v = opt.standardize(data)
with print_context(formatter={'float': '{: 1.4f}'.format}):
    X = data_m_v
    print('mean:')
    print(X.mean(axis=0))
    print('covariance:')
    print(np.cov(X.T))

# new estimates of mean and variance-covariance matrices
new_mean = np.array([0.005, 0.006])
new_cov = np.array([[0.002025, 0.00198],[0.00198, 0.003025]])
new_data = opt.rescale(data_m_v, new_mean, new_cov)

# print its mean and variance-covariance
with print_context(formatter={'float': '{: 1.5f}'.format}):
    X = new_data
    print('mean:\n', X.mean(axis=0))
    print('covariance:\n', np.cov(X.T))
    
# use Cornish-Fisher formulas to compute Value-at-Risk (VaR)
X = new_data[:,0]
Y = new_data[:,1]
quantile_x = opt.approx_quantile(X, q=0.05)
quantile_y = opt.approx_quantile(Y, q=0.05)
print('VaR for X: {:1.4f}'.format(quantile_x))
print('VaR for Y: {:1.4f}'.format(quantile_y))
    
# Compute Expected Shortfall or Conditional VaR
cvar_x = 0
cvar_y = 0
for i in range(5):
    q = (i + 1)/100    
    cvar_x += opt.approx_quantile(X, q=q)
    cvar_y += opt.approx_quantile(Y, q=q)
cvar_x = cvar_x/5
cvar_y = cvar_y/5
    
print('CVaR for X: {:1.4f}'.format(cvar_x))
print('CVaR for Y: {:1.4f}'.format(cvar_y))

# optimize a portfolio with this assets
initial = np.array([0.5,0.5])
res=opt.minimize_cvar(new_data,0.05,initial,display=True)

print('Optimal weights: {}'.format(res.x))
print('Minimum CVaR: {}'.format(res.fun))