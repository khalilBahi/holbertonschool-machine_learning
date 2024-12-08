#!/usr/bin/env python3
"""
This module provides a function to calculate the shape of a NumPy array.
The `np_shape` function returns the shape of a
given NumPy array as a tuple of integers.
"""


def np_shape(matrix):
    """
    Calculate the shape of a NumPy array.

    Args:
        matrix (numpy.ndarray): The NumPy array
        for which to calculate the shape.

    Returns:
        tuple: A tuple of integers representing the shape of the array.
    """
    return matrix.shape
