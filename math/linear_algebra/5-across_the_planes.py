#!/usr/bin/env python3
"""
This module provides a function to perform
element-wise addition of two 2D matrices.

The `add_matrices2D` function takes two 2D matrices (lists of lists of numbers)
and returns a new 2D matrix where each element is the sum of the corresponding
elements in the input matrices. If the matrices do not have the same dimensions
the function returns None.

Example:
    >>> add_matrices2D([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[6, 8], [10, 12]]
    >>> add_matrices2D([[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]])
    None
"""


def add_matrices2D(mat1, mat2):
    """
    Perform element-wise addition of two 2D matrices.

    The function takes two 2D matrices and returns a new matrix where each
    element is the sum of the corresponding elements in the input matrices.
    If the matrices do not have the same dimensions, it returns None.

    Args:
        mat1 (list of lists of int/float): The first 2D matrix.
        mat2 (list of lists of int/float): The second 2D matrix.

    Returns:
        list of lists of int/float: A new 2D matrix
        with the element-wise sums of the inputs,
        or None if the matrices do not have the same dimensions.

    """
    # Check if matrices have the same dimensions
    if len(mat1) != len(mat2) or any(len(row1) != len(row2)
                                     for row1, row2 in zip(mat1, mat2)):
        return None
