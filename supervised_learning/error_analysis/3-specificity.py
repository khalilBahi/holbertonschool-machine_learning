#!/usr/bin/env python3
"""Task 3: 3. Specificity"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity (true negative rate)
    for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)

    Returns:
        numpy.ndarray: Specificity for each class, of shape (classes,).
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    specificity = TN / (TN + FP)
    return specificity
