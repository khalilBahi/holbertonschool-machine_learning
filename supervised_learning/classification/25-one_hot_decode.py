#!/usr/bin/env python3
"""25. One-Hot Decode"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    one_hot is a one-hot numpy.ndarray with shape (classes, m)
    Returns: a numpy.ndarray with shape (m,) containing the numeric labels,
    or None on failure
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    return np.array([np.where(i == 1)[0][0] for i in one_hot.T])
