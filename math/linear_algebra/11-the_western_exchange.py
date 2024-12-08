#!/usr/bin/env python3
"""
This module provides a function to transpose a NumPy array.
The `np_transpose` function returns the transposed
version of a given NumPy array.
"""


def np_transpose(matrix):
    """
    Transpose a NumPy array.

    Args:
        matrix (numpy.ndarray): The NumPy array to transpose.

    Returns:
        numpy.ndarray: A new NumPy array that is
        the transpose of the input matrix.
    """
    return matrix.T
