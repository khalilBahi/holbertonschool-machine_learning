#!/usr/bin/env python3
"""Task 0: 0. Normalization Constants"""
import numpy as np


def normalization_constants(X):
    """
    Calculate the mean and standard deviation of each feature in the matrix X.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx) where
        m is the number of data points
        and nx is the number of features.

    Returns:
        mean (numpy.ndarray): Mean of each feature, shape (nx,).
        std (numpy.ndarray): Standard deviation of each feature, shape (nx,).
    """
    # Calculate the mean along the columns (axis=0)
    mean = np.mean(X, axis=0)
    # Calculate the standard deviation along the columns (axis=0)
    std = np.std(X, axis=0)
    return mean, std
