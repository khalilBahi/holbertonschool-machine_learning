#!/usr/bin/env python3
"""Task 1: 1. Normalize"""
import numpy as np


def normalize(X, m, s):
    """
    Normalize (standardize) a matrix X using the mean and standard deviation.

    Args:
        X (numpy.ndarray): Matrix of shape (d, nx) where d is the number of data points
                           and nx is the number of features.
        m (numpy.ndarray): Mean of each feature, shape (nx,).
        s (numpy.ndarray): Standard deviation of each feature, shape (nx,).

    Returns:
        numpy.ndarray: The normalized matrix X.
    """
    # Normalize X by subtracting the mean and dividing by the standard
    # deviation
    X_normalized = (X - m) / s
    return X_normalized
