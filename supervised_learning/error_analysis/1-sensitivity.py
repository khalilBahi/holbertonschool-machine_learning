#!/usr/bin/env python3
"""Task 1: 1. Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray):
        Confusion matrix of shape (classes, classes).

    Returns:
        numpy.ndarray: Sensitivity for each class, of shape (classes,).
    """
    # True Positives (TP) are the diagonal elements
    true_positives = np.diag(confusion)

    # False Negatives (FN) are the sum of the row minus the diagonal element
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # Sensitivity = TP / (TP + FN)
    sensitivity = true_positives / (true_positives + false_negatives)

    return sensitivity
