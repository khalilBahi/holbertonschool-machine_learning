#!/usr/bin/env python3
"""Task 2: 2. Precision"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): A confusion matrix of shape
    (classes, classes) where row indices represent the correct labels
    and column indices represent the predicted labels.

    Returns:
    numpy.ndarray: An array of shape (classes,) containing the precision
    of each class.
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
