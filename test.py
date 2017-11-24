# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:39:33 2017

@author: Jesus
"""

import pandas as pd
import numpy as np
import cvar_optimization as opt
import matplotlib.pyplot as pp
from print_context import print_context

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
    
# Compute Expected Shortfall or Conditional VaR using Cornish-Fisher approx.
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

# make fake data
np.random.seed(0)
fake_mean = np.array([0.05, 0.01])
fake_variance = np.array([[0.0025, 0.0006],[0.0006, 0.0009]])
data = np.random.multivariate_normal(mean=fake_mean,cov=fake_variance,size=1000)

# optimize a portfolio using Markowitz framework
min_return = 0.02
initial = np.array([0.5,0.5])
res = opt.mean_variance_optimization(data, min_return=min_return, initial=initial)

print('Optimal weights: ({0:1.4f}, {1:1.4f})'.format(res.x[0], res.x[1]))
print('Optimal Return: {:1.4f}'.format(res.x@fake_mean))
print('Optimal Risk: {:1.4f}'.format(np.sqrt(res.fun)))

# compute the efficient frontier and plot it
returns, risks , opt_weights = opt.efficient_frontier(data)
pp.plot(risks, returns)
pp.title('Efficient Frontier')
pp.xlabel('Risk')
pp.ylabel('Return')

# optimize a portfolio using mean-cvar framework
min_return = 0.02
res = opt.mean_cvar_optimization(data, min_return=min_return)

print('Optimal weights: ({0:1.4f}, {1:1.4f})'.format(res.x[0], res.x[1]))
print('Optimal Return: {:1.4f}'.format(res.x@fake_mean))
print('Optimal Risk: {:1.4f}'.format(np.sqrt(res.fun)))