# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:39:33 2017

@author: Jesus
"""

import pandas as pd
import numpy as np

from contextlib import contextmanager
@contextmanager
def print_context(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)
        
import cvar_optimization as opt

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

# optimize a portfolio with this assets
initial = np.array([0.5,0.5])
res=opt.minimize_cvar(new_data,0.05,initial,display=True)

print('Optimal weights: {}'.format(res.x))
print('Minimum CVaR: {}'.format(res.fun))