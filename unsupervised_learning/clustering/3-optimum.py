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
    # Basic validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and not isinstance(kmax, int):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, _ = X.shape
    # Determine effective kmax and cap to n-1 to satisfy variance constraints
    eff_kmax = (kmax if kmax is not None else n - 1)
    eff_kmax = min(eff_kmax, n - 1)

    # Must analyze at least two different cluster sizes
    if eff_kmax < kmin + 1:
        return None, None

    results = []
    d_vars = []
    base_variance = None

    # Single loop over cluster sizes [kmin, eff_kmax]
    for k in range(kmin, eff_kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        var_k = variance(X, C)
        if var_k is None:
            return None, None
        if base_variance is None:
            base_variance = var_k
            d_vars.append(0.0)
        else:
            d_vars.append(base_variance - var_k)

    return results, d_vars
