#!/usr/bin/env python3
"""
This module provides a function for concatenating two NumPy arrays
along a specified axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two NumPy arrays along a specified axis.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray): The second array.
        axis (int): The axis along which to concatenate. Default is 0.

    Returns:
        numpy.ndarray: A new array resulting from concatenation.
    """
    return np.concatenate((mat1, mat2), axis=axis)
