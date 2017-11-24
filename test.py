# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:39:33 2017

@author: Jesus
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as pp
from portfolio_optimization import mean_cvar_optimization, efficient_frontier_cvar
from print_context import print_context

# import data
raw_data = pd.read_excel('test_data.xlsx',sheetname='data_sample')
raw_data = raw_data.iloc[:,1:9]
raw_data = raw_data.values

# boostraping data
np.random.seed(0)
num_obs = len(raw_data)
indexes = np.random.choice(num_obs,1000)
data = np.array([raw_data[index] for index in indexes])

# print its mean, variance-covariance, skewness and kurtosis
with print_context(formatter={'float': '{: 1.4f}'.format}):
    X = data
    print('expected returns:')
    print(np.mean(X, axis=0))
    print('covariance:')
    print(np.cov(X.T))
    print('skewness:')
    print(skew(X))
    print('(excess) kurtosis:')
    print(kurtosis(X))

# optimize a portfolio using mean-cvar framework
min_return = 0.04
beta = 0.95
res = mean_cvar_optimization(data, min_return=min_return, beta=beta)

# print optimal weights
with print_context(formatter={'float': '{: 1.4f}'.format}):
    print('Optimal weights:')
    print(res.x)

# expected return and covariance matrix
expected_return = np.mean(data,axis=0)
covariance = np.cov(data.T)

print('Optimal Return: {:1.4f}'.format(res.x@expected_return))
print('Optimal Risk (CVaR): {:1.4f}'.format(res.fun))
print('Risk (Std. Dev.): {:1.4f}'.format(np.sqrt(res.x@covariance@res.x)))

# compute the efficient frontier and plot it
num_points = 40
beta = 0.95
returns, risks , cvars, weights = efficient_frontier_cvar(data, beta=beta, 
                                                              num_points=num_points)
pp.plot(risks, returns)
pp.title('Efficient Frontier')
pp.xlabel('Risk')
pp.ylabel('Return')

# collecting data to write it in Excel
results = np.hstack([weights, returns.reshape(num_points, 1), 
                risks.reshape(num_points, 1), cvars.reshape(num_points, 1)])

columns = ['Asset ' + str(i + 1) for i in range(len(expected_return))]
columns = columns + ['Return', 'Std. Dev.', 'CVaR']
df = pd.DataFrame(results, columns=columns)
pd.options.display.float_format = '{:,.4f}'.format
print(df)

writer = pd.ExcelWriter('results.xlsx')
df.to_excel(writer,sheet_name='mean_cvar')