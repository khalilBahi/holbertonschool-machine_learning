#!/usr/bin/env python3
"""Task 4: 4. F1 score"""
import numpy as np


def f1_score(confusion):
    """
    Calculate the F1 score for each class from a confusion matrix.

    Args:
    confusion (numpy.ndarray): A confusion matrix of shape (classes, classes).

    Returns:
    numpy.ndarray: An array of shape (classes,)
    containing the F1 score of each class.
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    sens = sensitivity(confusion)
    prec = precision(confusion)

    # Calculate F1 score for each class
    F1 = 2 * (prec * sens) / (prec + sens)

    return F1
