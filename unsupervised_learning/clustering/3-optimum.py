#!/usr/bin/env python3
"""Optimize k for K-means clustering"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Parameters:
    - X (numpy.ndarray): 2D numpy array of shape (n, d) containing the dataset.
    - kmin (int): Minimum number of clusters to check for (inclusive).
    - kmax (int): Maximum number of clusters to check for (inclusive).
    - iterations (int): Maximum number of iterations for K-means.

    Returns:
    - tuple: (results, d_vars), or (None, None) on failure.
        - results is a list containing the outputs of K-means for each
        cluster size.
        - d_vars is a list containing the difference in variance from the
        smallest cluster size for each cluster size.
    """
    # Validation matching reference implementation
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is not None and (type(kmax) is not int or kmax < 1):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    results = []
    d_vars = []
    
    # Loop over cluster sizes [kmin, kmax] inclusive
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        # Guard against None returns to prevent TypeError
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        
        var = variance(X, C)
        # Guard against None variance
        if var is None:
            return None, None
            
        if k == kmin:
            first_var = var
        d_vars.append(first_var - var)

    return results, d_vars
