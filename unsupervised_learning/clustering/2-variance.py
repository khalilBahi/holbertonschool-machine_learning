#!/usr/bin/env python3
"""
2. Variance
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        C: numpy.ndarray of shape (k, d) containing the centroid means

    Returns:
        var: total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        # print("Invalid X, must be np.ndarray of shape(n, d)")
        return None
    n, dx = X.shape
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        # print("Invalid X, must be np.ndarray of shape(n, d)")
        return None
    k, dc = C.shape
    if not isinstance(k, int) or k <= 0 or k >= n or dx != dc:
        # print("Invalid k, must be int > 0 and < n")
        return None
    # sqrt((x1 - X2)^2 + (y1 - y2)^2)
    dist = ((X - C[:, np.newaxis]) ** 2).sum(axis=2)
    min_dist = np.min(dist, axis=0)
    # print(min_dist.sum())
    # print(min_dist.shape)
    var = np.sum(min_dist)
    # print(var)
    return var
