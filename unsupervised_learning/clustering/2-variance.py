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
    # Input validation
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2 or X.shape[1] != C.shape[1]:
        return None

    # Calculate distances to all centroids for all points
    # X: (n, d), C[:, np.newaxis]: (k, 1, d) -> distances: (k, n, d)
    distances = ((X - C[:, np.newaxis]) ** 2).sum(axis=2)

    # Get minimum distance for each point (distance to its cluster centroid)
    min_distances = np.min(distances, axis=0)

    # Total variance is sum of squared distances to nearest centroid
    var = np.sum(min_distances)

    return var
