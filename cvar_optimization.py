# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 07:39:13 2017

@author: Jesus
"""

import numpy as np
from scipy.stats import norm, moment
from scipy.optimize import minimize

def standardize(data):
    """
    Return a standarized dataset given data
    
    Parameters
    ----------
        data: array_like of shape(M, N) where N > 1
            
    Return
    ------
        out: array_like with same shape of data
        
    """
    # remove the mean
    data = data - data.mean(axis=0)
    # scale the covariance
    cov = np.cov(data.T)
    chol = np.linalg.cholesky(cov)
    inv_chol = np.linalg.inv(chol)
    return data@inv_chol.T

def rescale(data, new_mean, new_cov):
    """
    Return a rescale dataset given a standarized data new means and covariance 
    matrix
    
    Parameters
    ----------
        data: array_like of shape(M, N) where N > 1
        new_mean: array_like of shape(N, )
        new_cov: array_like of shape(N, N)
    
    Return
    ------
        out: array_like with same shape of data
        
    """
    # rescale the covariance using the new covariance
    chol = np.linalg.cholesky(new_cov)
    data = data@chol.T
    # add the new mean
    return data + new_mean

def approx_quantile(X, q):
    """
    Compute an approximation of the q quantile of X using Cornish-Fisher 
    expansion
    
    Parameters
    ----------
        X: array_like of shape(M, )
        q: scalar between 0 and 1
    
    Return
    ------
        out: scalar
    
    """
    # Compute the cumulants
    
    k1 = np.mean(X)
    k2 = moment(X, moment=2)
    k3 = moment(X, moment=3)
    k4 = moment(X, moment=4) - 3*k2**2
    k5 = moment(X, moment=5) - 10*k3*k2
    z = cornish_fisher_expansion(q, k1, k2, k3, k4, k5)
    return z*k2**0.5 + k1

def cornish_fisher_expansion(q, k1, k2, k3, k4, k5):
    """
    Compute Cornish-Fisher expansion based on its five first cumulants
    
    Parameters
    ----------
        q: scalar between 0 and 1
        k: scalar
    
    Return
    ------
        out: scalar
    
    """
    z = norm.ppf(q) # desired quantile
    x_q = z + (z**2 - 1)/6*k3 + (z**3 - 3*z)/24*k4 - (2*z**3 - 5*z)/36*k3**2
    x_q += (z**4 - 6*z**2 + 3)/120*k5 - (z**4 - 5*z**2 + 2)/24*k3*k4 + (12*z**4 - 53*z**2 + 17)/324*k3**3
    return x_q


# objective function
def minimize_cvar(new_data, quantile, initial, method='SLSQP', display=False):
    """
    Compute optimal weights after minimizing the cvar for given a quantile 
    level
    
    Parameters
    ----------
        new_data: array_like of shape(M, N) with N > 1
        quantile: scalar between 0 and 1
        initial: array_like of shape(N, )
    
    Return
    ------
        out: OptimizeResult object that stores the optimal value and weight
            among other features after optimization
    
    """
    def func(weights):
        """
        Objective function value to optimize
        """
        portfolio_returns = new_data@weights
        return - approx_quantile(portfolio_returns, quantile)
    
    # constraints
    def const_1(weights):
        """
        Contraint 1: Weights add up 1
        """
        return np.sum(weights) - 1
    
    def const_2(weights):
        """
        Contraint 2: Weights are non-negative
        """        
        return weights
    
    def const_3(weights):
        """
        Contraint 3: Weights are less than 1
        """
        return 1 - weights
    
    # setting up constraints as dictionary
    cons = ({'type': 'eq',
            'fun': lambda x: const_1(x)},
           {'type': 'ineq',
           'fun': lambda x: const_2(x)},
           {'type': 'ineq',
           'fun': lambda x: const_3(x)},)
    
    if method=='SLSQP':
        # Sequential Least SQuares Programming (SLSQP)
        return minimize(func, initial, constraints=cons, method='SLSQP', options={'disp': True})
    else:
        print('Please, introduce a valid method')
        return None