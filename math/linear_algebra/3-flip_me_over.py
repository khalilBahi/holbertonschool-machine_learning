#!/usr/bin/env python3
"""
This module provides a function to calculate the transpose of a matrix.

The `matrix_transpose` function takes a matrix (represented as a list of lists)
and returns its transpose, where rows are converted to columns and vice versa.

Example:
    >>> matrix_transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    >>> matrix_transpose([[1, 2], [3, 4], [5, 6]])
    [[1, 3, 5], [2, 4, 6]]
"""


def matrix_transpose(matrix):
    """
    Calculate the transpose of a matrix.

    The function returns a new matrix that is the transpose of the input matrix,
    meaning its rows are converted to columns.

    Args:
        matrix (list of lists): A 2D list representing the matrix to be transposed.

    Returns:
        list of lists: A new 2D list representing the transposed matrix.
    """
    return [list(row) for row in zip(*matrix)]
