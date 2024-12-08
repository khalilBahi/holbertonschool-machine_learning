#!/usr/bin/env python3
"""
This module provides a function for performing matrix multiplication.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication of two numpy arrays.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The result of matrix multiplication.
    """
    return np.matmul(mat1, mat2)
