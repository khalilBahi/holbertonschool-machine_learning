#!/usr/bin/env python3
"""Task 13: 13. Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural
    network using batch normalization.

    Args:
        Z (numpy.ndarray): Input matrix of shape (m, n).
        gamma (numpy.ndarray): Scale parameter of shape (1, n).
        beta (numpy.ndarray): Offset parameter of shape (1, n).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized Z matrix.
    """
    # Calculate the mean and variance of Z along the batch axis (axis 0)
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    # Normalize Z
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift using gamma and beta
    Z_normalized = gamma * Z_normalized + beta

    return Z_normalized
