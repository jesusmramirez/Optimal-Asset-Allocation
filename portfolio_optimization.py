# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 07:39:13 2017

@author: Jesus
"""

import numpy as np
from scipy.stats import norm, moment
from scipy.optimize import minimize, linprog
from functools import partial

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
    data = data - np.mean(data, axis=0)
    # scale the covariance
    cov = np.cov(data.T, ddof=0)
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

# minimum-VaR portfolio given a minimum portfolio return (Markowitz optimization)
def mean_mvar_optimization(data, min_return, initial, quantile=0.05, 
                           expected_returns=None, display=False):
    """
    Compute optimal weights after minimizing the cvar for given a quantile 
    level
    
    Parameters
    ----------
        data: array_like of shape(M, N) with N > 1
        min_return: scalar float
            specifies the minimum return that the portfolio can attain
        initial: array_like of shape(N, )
        quantile: scalar between 0 and 1, default value 0.05
        expected_returns: array_like of shape(N, )
        display: boolean, default value is False
            if it is True, a summary of the optimization procedure is shown
    
    Return
    ------
        out: OptimizeResult object that stores the optimal value and weight
            among other features after optimization
    
    """
    if expected_returns is None:
        expected_returns = np.mean(data, axis=0)

    def func(weights):
        """
        Objective function value to optimize
        """
        portfolio_returns = data@weights
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

    def const_4(weights):
        """
        Contraint 4: portfolio return can be lower than a given fixed minimum 
        return
        """
        return weights@expected_returns - min_return
    
    # setting up constraints as dictionary
    cons = ({'type': 'eq',
            'fun': lambda x: const_1(x)},
           {'type': 'ineq',
           'fun': lambda x: const_2(x)},
           {'type': 'ineq',
           'fun': lambda x: const_3(x)},
           {'type': 'ineq',
           'fun': lambda x: const_4(x)})

    # Sequential Least SQuares Programming (SLSQP)
    res = minimize(func, initial, constraints=cons, method='SLSQP', 
                   options={'disp': display})
    return res


# minimum-variance portfolio given a minimum portfolio return (Markowitz optimization)
def mean_variance_optimization(min_return, initial, data=None, expected_returns=None, 
                               covariance=None, display=False):
    """
    Computes the optimal weights that minimize portfolio variance subject to minimum 
    level of return in the portfolio using Sequential Least SQuares Programming
    
    Parameters
    ----------
        min_return: scalar float
            specifies the minimum return that the portfolio can attain
        initial: array_like of shape(N, )
        data: array_like of shape(M, N) with N > 1
        expected_returns: array_like of shape (N, )
        covariance: array_like of shape (N, N)
        display: boolean, default value is False
            if it is True, a summary of the optimization procedure is shown

    Return
    ------
        out: OptimizeResult object that stores the optimal value and weights
            among other features after optimization
            (for more information see scipy.optimize documentation)
    
    """
    if expected_returns is None:
        expected_returns = np.mean(data, axis=0)

    if covariance is None:
        covariance = np.cov(data.T, ddof=0)
    
    def func(weights):
        """
        Objective function value to optimize
        """
        return weights@covariance@weights

    # constraints
    def const_1(weights):
        """
        Contraint 1: weights add up 1
        """
        return np.sum(weights) - 1

    def const_2(weights):
        """
        Contraint 2: weights are non-negative
        """     
        return weights

    def const_3(weights):
        """
        Contraint 3: weights are never greater than 1
        """
        return 1 - weights
    
    def const_4(weights):
        """
        Contraint 4: portfolio return can be lower than a given fixed minimum 
        return
        """
        return weights@expected_returns - min_return
    
    # setting up constraints as dictionary
    cons = ({'type': 'eq',
            'fun': lambda x: const_1(x)},
           {'type': 'ineq',
           'fun': lambda x: const_2(x)},
           {'type': 'ineq',
           'fun': lambda x: const_3(x)},
           {'type': 'ineq',
           'fun': lambda x: const_4(x)})
    
    # Sequential Least SQuares Programming (SLSQP)
    res = minimize(func, initial, constraints=cons, method='SLSQP', 
                   options={'disp': display})
    return res


def mean_cvar_optimization(data, min_return, beta=0.95, expected_returns=None, 
                           display=False):
    """
    Computes the optimal weights that minimize beta-CVaR subject to minimum 
    level of return in the portfolio using linear programming (simplex method)
    
    Parameters
    ----------
        data: array_like of shape (M, N) where M denotes the number of 
            draws and N the number of assets each row is a draw from 
            joint distribution of returns
        min_return: scalar float
            specifies the minimum return that the portfolio can attain
        beta: scalar between 0 and 1, common values for beta are 0.9, 0.95 or 0.99
        expected_returns: array_like of shape (N, )

    Return
    ------
        out: OptimizeResult object that stores the optimal value and weights
            among other features after optimization
            (for more information see scipy.optimize documentation)
    
    References
    ----------
        Rockafellar and Uryasev, Optimization of Conditional Value-At-Risk. The 
        Journal of Risk, Vol. 2, No. 3, 2000, 21-41 
    
    """

    if expected_returns is None:
        expected_returns = np.mean(data, axis=0)

    # for this version I keep Rockafellar and Uryasev's notation
    n = data.shape[1]
    q = data.shape[0]
    y = data
    m = expected_returns
    R = min_return
    
    # set minimization problem
    
    # define vector for the objective function
    c = np.hstack([np.zeros(n), np.ones(q)/((1 - beta)*q), 1])
    
    # define matrices for equality constraints
    A_eq = np.hstack([np.ones(n), np.zeros(q + 1)]).reshape(1, n + q + 1)
    b_eq = np.ones(1)
    
    # define matrices for inequality constraints
    A_ub = np.hstack([-m, np.zeros(q + 1)])
    A_ub = np.vstack([A_ub, np.hstack([-y, -np.eye(q), -np.ones(q).reshape(q, 1)])])
    b_ub = np.hstack([-R, np.zeros(q)])
    
    # define a sequence for bounds
    bounds = tuple([(0, None) for _ in range(n + q)] + [(None, None)])
    
    # minimize beta-CVaR
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    res.x = res.x[:n]
    return res


def resampling_optimization(data, min_return, initial, sample_size=100, 
                            expected_returns=None, covariance=None, display=False):
    """
    Computes the optimal weights that minimize beta-CVaR subject to minimum 
    level of return in the portfolio using linear programming (simplex method)
    
    Parameters
    ----------
        data: array_like of shape (M, N) where M denotes the number of 
            draws and N the number of assets each row is a draw from 
            joint distribution of returns
        min_return: scalar float
            specifies the minimum return that the portfolio can attain
        initial: array_like of shape(N, )
        sample_size: int scalar, default value 100
            specifies the sample size for resampling
        expected_returns: array_like of shape (N, )
        covariance: array_like of shape (N, N)
        display: boolean, default value is False
            if it is True, a summary of the optimization procedure is shown


    Return
    ------
        out: OptimizeResult object that stores the optimal value and weights
            among other features after optimization
            (for more information see scipy.optimize documentation)
    
    References
    ----------
        Rockafellar and Uryasev, Optimization of Conditional Value-At-Risk. The 
        Journal of Risk, Vol. 2, No. 3, 2000, 21-41 
    
    """
    scale = 0.1
    variances = np.var(data, axis=0)*scale
    num_asset = len(variances)
    shocks = np.random.normal(scale=variances, size=(sample_size, num_asset))
    
    weights =  np.zeros(shape=(sample_size, num_asset))
    for i in range(sample_size):
        shocked_data = data + shocks[i]
        res = mean_variance_optimization(shocked_data, min_return, initial, 
                                         display=display)
        weights[i]= res.x
        
    return  weights


def efficient_frontier(data=None, expected_returns=None, covariance=None, num_points=10):
    """
    Computes the efficient frontier
    
    Parameters
    ----------
        data: array_like of shape(M, N) where M denotes the number of 
            draws and N the number of assets each row is a draw from 
            joint distribution of returns
        num_points: number of points for the efficient frontier

    Return
    ------
        out: 3 array_like of shape(num_points, ) as follows: portfolio returns,
            portfolio risks and optimal weights
    
    """
    # compute expected return and covariance matrix from data if they are not 
    # passed
    if expected_returns is None:
        expected_returns = np.mean(data, axis=0)

    if covariance is None:
        covariance = np.cov(data.T, ddof=0)
    
    # create a parcial function from mean_variance that assumes given the expected
    # returns and covariance matrix
    markowitz_optimization = partial(mean_variance_optimization, 
                                     expected_returns=expected_returns, 
                                     covariance=covariance)
    
    # make a grid for minimum portfolio return (short-selling is not allowed)
    lower_return = min(expected_returns)
    upper_return = max(expected_returns)
    min_returns = np.linspace(lower_return, upper_return, num=num_points)

    # initialize weights for portfolio optimization
    num_assets = len(expected_returns)    
    
    weights = np.zeros(shape=(num_points, num_assets))
    initial = np.ones(num_assets)/num_assets
    
    for i, min_return in enumerate(min_returns):
        res = markowitz_optimization(min_return=min_return, initial=initial)
        weights[i] = res.x[:num_assets]
        
    # compute the efficient frontier
    # porfolio return for each optimal portfolio
    portfolio_return = weights@expected_returns
    portfolio_risk = np.zeros_like(portfolio_return)
    
    for i in range(num_points):
        # portfolio risk (standard deviation) for each optimal portfolio
        portfolio_risk[i] = np.sqrt(weights[i]@covariance@weights[i])
    
    return portfolio_return, portfolio_risk, weights


def efficient_frontier_cvar(data, beta=0.95, num_points=10):
    """
    Computes the efficient frontier
    
    Parameters
    ----------
        data: array_like of shape(M, N) where M denotes the number of 
            draws and N the number of assets each row is a draw from the joint 
            distribution of returns
        beta: scalar between 0 and 1, common values for beta are 0.9, 0.95 or 0.99
        num_points: number of points for the efficient frontier

    Return
    ------
        out: 3 array_like of shape(num_points, ) as follows: portfolio returns,
            portfolio risks and optimal weights
    
    """
    # compute expected return and covariance matrix from data
    expected_returns = np.mean(data, axis=0)
    covariance = np.cov(data.T, ddof=0)
    
    # make a grid for minimum portfolio return (short-selling is not allowed)
    lower_return = min(expected_returns)
    upper_return = max(expected_returns)
    min_returns = np.linspace(lower_return, upper_return, num=num_points)

    # initialize weights for portfolio optimization
    num_assets = len(expected_returns)
    
    weights = np.zeros(shape=(num_points, num_assets))
    cvars = np.zeros(shape=num_points)
    for i, min_return in enumerate(min_returns):
        res = mean_cvar_optimization(data, min_return, beta)
        weights[i] = res.x[:num_assets]
        cvars[i] = res.fun        
        
    # compute the efficient frontier
    # porfolio return for each optimal portfolio
    portfolio_return = weights@expected_returns
    portfolio_risk = np.zeros_like(portfolio_return)
    
    for i in range(num_points):
        # portfolio risk (standard deviation) for each optimal portfolio
        portfolio_risk[i] = np.sqrt(weights[i]@covariance@weights[i])
    
    return portfolio_return, portfolio_risk, cvars, weights


def efficient_frontier_resampling(data=None, expected_returns=None, covariance=None, 
                                  num_bins=10, sample_size=100, scale=0.1):
    """
    Computes the efficient frontier
    
    Parameters
    ----------
        data: array_like of shape(M, N) where M denotes the number of 
            draws and N the number of assets each row is a draw from 
            joint distribution of returns
        num_points: number of points for the efficient frontier
        sample_size: int scalar, default value 100
            specifies the size for each sample in the resampling procedure
        num_bins: int scalar, default value 10
            specifies the number of bins to which the portfolios will be group

    Return
    ------
        out: 3 array_like of shape(num_points, ) as follows: portfolio returns,
            portfolio risks and optimal weights
    
    """
    # compute expected return and covariance matrix from data if they are not 
    # passed
    if expected_returns is None:
        expected_returns = np.mean(data, axis=0)

    if covariance is None:
        covariance = np.cov(data.T, ddof=0)
    
    # generate random normal shocks with mean zero and variance 1/10th of its 
    # std. deviation
    std = np.std(data, axis=0)*scale
    num_assets = len(std)
    noise = np.random.normal(scale=np.square(std), size=(sample_size, num_assets))

    # initialize weights for portfolio optimization    
    r_weights = np.zeros(shape=(num_bins*sample_size, num_assets))    
    
    # measure the risk for each portfolio
    portfolio_risk = np.zeros(shape=num_bins*sample_size)
    
    # compute the efficient frontier for each sample of noised expected returns
    for j in range(sample_size):
        noised_expected_returns = expected_returns + noise[j]
        returns, risks, weights  = efficient_frontier(expected_returns=noised_expected_returns, 
                                                      covariance=covariance, num_points=num_bins)
        r_weights[j*num_bins: (j + 1)*num_bins] = weights
        portfolio_risk[j*num_bins: (j + 1)*num_bins] = risks
    
    # group portfolio weights by bins of same number of elements, average them 
    # and normalize to one
    index = np.argsort(portfolio_risk)
    sorted_weights = r_weights[index]
    
    average_weights = []
    for j in range(num_bins):
        average_weight = sorted_weights[sample_size*j:sample_size*(j + 1),:]
        average_weight = np.sum(average_weight, axis=0)/np.sum(average_weight)
        average_weights.append(average_weight)
    
    average_weights = np.array(average_weights)
    # compute the efficient frontier
    # porfolio return for each optimal portfolio
    portfolio_return = average_weights@expected_returns
    portfolio_risk = np.zeros_like(portfolio_return)
    for i in range(num_bins):
        # portfolio risk (standard deviation) for each optimal portfolio
        portfolio_risk[i] = np.sqrt(average_weights[i]@covariance@average_weights[i])

    return portfolio_return, portfolio_risk, average_weights