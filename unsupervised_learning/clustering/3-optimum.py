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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or not isinstance(iterations, int):
        return None, None
    if kmin < 1 or iterations < 1:
        return None, None

    n = X.shape[0]
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin or kmax > n:
        return None, None
    if kmax - kmin < 1:  # Ensure at least 2 clusters are tested
        return None, None

    # Initialize lists for results
    results = []
    variances = []

    # First loop: run K-means for each k
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:  # Check for kmeans failure
            return None, None
        results.append((C, clss))
        var = variance(X, C)
        if var is None:  # Check for variance failure
            return None, None
        variances.append(var)

    # Calculate differences from minimum cluster size variance
    base_var = variances[0]  # Variance at kmin
    d_vars = [base_var - var for var in variances]

    return results, d_vars
