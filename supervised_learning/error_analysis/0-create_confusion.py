#!/usr/bin/env python3
"""Task 0: 0. Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from the given labels and logits.

    Parameters:
    labels (numpy.ndarray): One-hot encoded array of shape
    (m, classes) containing the correct labels.
    logits (numpy.ndarray): One-hot encoded array of shape
    (m, classes) containing the predicted labels.

    Returns:
    numpy.ndarray: Confusion matrix of shape
    (classes, classes) with row indices representing the correct labels
    and column indices representing the predicted labels.
    """
    return np.dot(labels.T, logits)
