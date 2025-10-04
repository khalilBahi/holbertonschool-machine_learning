#!/usr/bin/env python3
"""
1. K-means
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer number of clusters
        iterations: positive integer maximum number of iterations

    Returns:
        C, clss: tuple containing centroids and cluster assignments,
        or None, None on failure
    """
    # Input validation
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(k, int)
        or not isinstance(iterations, int)
    ):
        return None, None
    if len(X.shape) != 2 or k <= 0 or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    # Main iteration loop
    for _ in range(iterations):
        # Calculate distances and assign clusters
        distances = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        # Store previous centroids for convergence check
        C_prev = C.copy()

        # Update centroids
        for i in range(k):
            cluster_points = X[clss == i]
            if len(cluster_points) > 0:
                C[i] = np.mean(cluster_points, axis=0)
            else:
                # Reinitialize empty cluster
                C[i] = np.random.uniform(low=min_vals, high=max_vals, size=(d,))

        # Check for convergence
        if np.all(C == C_prev):
            break

    # Final cluster assignment
    distances = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
    clss = np.argmin(distances, axis=0)

    return C, clss
