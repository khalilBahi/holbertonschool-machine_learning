#!/usr/bin/env python3
"""Task 2: 2. Shuffle Data"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffle two matrices X and Y in the same way.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx)
        where m is the number of data points
                           and nx is the number of features in X.
        Y (numpy.ndarray): Matrix of shape (m, ny)
        where m is the same number of data points
                           as in X and ny is the number of features in Y.

    Returns:
        X[permutation] (numpy.ndarray): Shuffled version of X.
        Y[permutation] (numpy.ndarray): Shuffled version of Y.
    """
    # Generate a random permutation of indices
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
