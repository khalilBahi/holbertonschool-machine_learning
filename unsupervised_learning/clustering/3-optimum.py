#!/usr/bin/env python3
"""
3. Optimize k
"""
import numpy as np

kmeans = __import__("1-kmeans").kmeans
variance = __import__("2-variance").variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: minimum number of clusters to check (inclusive)
        kmax: maximum number of clusters to check (inclusive)
        iterations: maximum number of iterations for K-means

    Returns:
        results, d_vars: tuple containing K-means results and variance differences,
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None
    if not isinstance(iterations, int):
        return None, None
    if kmin < 1 or kmax < 1 or kmin >= kmax or iterations < 1:
        return None, None

    results, d_vars = [], []
    for i in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, i, iterations)
        results.append((centroids, clss))
        var = variance(X, centroids)
        if i == kmin:
            first_var = var
        # get diff
        d_vars.append(first_var - var)

    return results, d_vars
