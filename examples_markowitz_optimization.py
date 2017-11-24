# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:32:10 2017

@author: Jesus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from scipy.stats import skew, kurtosis
from portfolio_optimization import mean_variance_optimization, efficient_frontier
from print_context import print_context

# make data
np.random.seed(0)
mean = np.array([0.07, 0.03, 0.02])
corr = np.array([[1.0, 0.4, 0.1],[0.4, 1.0, 0.3],[0.1, 0.3, 1.0]])
std_dev = np.array([0.2, 0.07, 0.02])
cov = np.outer(std_dev, std_dev)*corr
data = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)

# expected return
expected_return = np.mean(data,axis=0)
covariance = np.cov(data.T)

# print its mean, variance-covariance, skewness and kurtosis
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


# optimize a portfolio using Markowitz framework
min_return = 0.025
initial = np.ones(shape=3)/3
res = mean_variance_optimization(data, min_return=min_return, initial=initial)

print('Optimal weights: ({0:1.4f}, {1:1.4f}, {2:1.4f})'.format(res.x[0], res.x[1], res.x[2]))
print('Optimal Return: {:1.4f}'.format(res.x@expected_return))
print('Optimal Risk (Std. Dev.): {:1.4f}'.format(np.sqrt(res.fun)))

# compute the efficient frontier and plot it
num_points = 20
returns, risks , weights = efficient_frontier(data, num_points=num_points)

pp.plot(risks, returns)
pp.title('Efficient Frontier')
pp.xlabel('Risk')
pp.ylabel('Return')

data = np.hstack([weights, returns.reshape(num_points, 1), risks.reshape(num_points, 1)])
columns = ['Asset 1','Asset 2', 'Asset 3', 'Return', 'Risk (Std. Dev.)']
df = pd.DataFrame(data, columns=columns)
pd.options.display.float_format = '{:,.5f}'.format
print(df)