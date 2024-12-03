#!/usr/bin/env python3
"""
This module provides a function to calculate the shape of a matrix.

The `matrix_shape` function determines the dimensions of a given matrix
(represented as nested lists), and returns its shape as a list of integers
where each integer represents the size of the matrix at a specific level
of nesting.

Example:
    >>> matrix_shape([[1, 2, 3], [4, 5, 6]])
    [2, 3]
    >>> matrix_shape([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    [2, 2, 2]
"""


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.

    The function determines the dimensions of a given matrix (nested lists),
    returning its shape as a list of integers. Each integer in the list
    represents the size of the matrix at a specific level of nesting.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list: A list of integers where each value represents the size of the
              corresponding dimension of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
