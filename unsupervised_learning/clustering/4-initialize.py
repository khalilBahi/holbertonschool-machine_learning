#!/usr/bin/env python3
"""
4. Initialize GMM
"""
import numpy as np

kmeans = __import__("1-kmeans").kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer number of clusters

    Returns:
        pi, m, S: tuple containing priors, centroids, and covariance matrices,
        or None, None, None on failure
    """
    # Input validation
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None, None, None
    if len(X.shape) != 2 or k <= 0:
        return None, None, None

    n, d = X.shape

    # Initialize priors evenly (1/k for each cluster)
    pi = np.ones(k) / k

    # Initialize centroids using K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # Initialize covariance matrices as identity matrices
    S = np.zeros((k, d, d))
    S[...] = np.eye(d)  # Broadcast identity matrix to all k clusters

    return pi, m, S
