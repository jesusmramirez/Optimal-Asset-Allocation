# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:54:00 2017

@author: Jesus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
from portfolio_optimization import efficient_frontier_resampling, efficient_frontier
from print_context import print_context

# import data
raw_data = pd.read_excel('test_data.xlsx',sheetname='data_sample')
raw_data = raw_data.iloc[:,1:9]
data = raw_data.values

# print its mean, variance-covariance, skewness and kurtosis
with print_context(formatter={'float': '{: 1.4f}'.format}):
    X = data
    print('expected returns:')
    print(np.mean(X, axis=0))
    print('covariance:')
    print(np.cov(X.T, ddof=0))

# compute the efficient frontier using resampling
num_points = 20
sample_size = 1000
num_bins = 20
scale = 0.7
r_returns, r_risks , r_weights = efficient_frontier_resampling(data, 
                                                             num_points=num_points, 
                                                             sample_size=sample_size,
                                                             num_bins=num_bins,
                                                             scale=scale)

# compute the efficient frontier
num_points = 20
returns, risks , weights = efficient_frontier(data, num_points=num_points)

# plot the efficient frontiers
pp.plot(r_risks, r_returns, '--k', label='Resampling')
pp.plot(risks, returns, '-r', label='Markowitz')
pp.title('Efficient Frontier')
pp.xlabel('Risk')
pp.ylabel('Return')
pp.legend(loc='lower right')

with print_context(formatter={'float': '{: 1.4f}'.format}):
    print('Average weights:')
    print(r_weights)
    print('Weights:')
    print(weights)
    
np.savetxt('r_weights.csv', r_weights, delimiter=',')
np.savetxt('weights.csv', weights, delimiter=',')
    
    
    