#!/usr/bin/env python3
"""24. One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Y is a numpy.ndarray with shape (m,) containing numeric class labels
    classes is the maximum number of classes

    Returns: one-hot encoding with shape (classes, m) or None on failure
    """
    if Y is None or type(Y) is not np.ndarray or type(classes) is not int:
        return None
    try:
        matrix = np.zeros((len(Y), classes))
        matrix[np.arange(len(Y)), Y] = 1
        return matrix.T
    except Exception:
        return None
