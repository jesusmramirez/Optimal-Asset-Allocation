# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:39:33 2017

@author: Jesus
"""

import pandas as pd
import numpy as np
import cvar_optimization as opt
import matplotlib.pyplot as pp

# import real data
data = pd.read_excel('test_data.xlsx',sheetname='data_sample')
data = data.iloc[:,[1,2]]/100
data = data.values

# boostraping data
np.random.seed(0)

# expected return
expected_return = np.mean(data,axis=0)
covariance = np.cov(data.T)

# optimize a portfolio using mean-cvar framework
min_return = 0.015
res = opt.mean_cvar_optimization(data, min_return=min_return, beta=0.95)

print('Optimal weights: ({0:1.4f}, {1:1.4f})'.format(res.x[0], res.x[1]))
print('Optimal Return: {:1.4f}'.format(res.x@expected_return))
print('Optimal Risk (CVaR): {:1.4f}'.format(res.fun))
print('Risk (Std. Dev.): {:1.4f}'.format(np.sqrt(res.x@covariance@res.x)))

# compute the efficient frontier and plot it
returns, risks , cvars, opt_weights = opt.efficient_frontier_cvar(data, beta=0.90)
pp.plot(risks, returns)
pp.title('Efficient Frontier')
pp.xlabel('Risk')
pp.ylabel('Return')

df = np.hstack([opt_weights, returns.reshape(10,1), cvars.reshape(10,1), risks.reshape(10,1)])
df = pd.DataFrame(df,columns=['Weight 1','Weight 2', 'Return', 'Risk (CVaR)', 'SD'])
pd.options.display.float_format = '{:,.8f}'.format
print(df)
writer = pd.ExcelWriter('results.xlsx')
df.to_excel(writer,sheet_name='mean_cvar')