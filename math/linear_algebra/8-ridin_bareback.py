#!/usr/bin/env python3
"""
This module provides a function for performing matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Perform matrix multiplication between two 2D matrices.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.

    Returns:
        list of list of int/float: The resulting matrix after multiplication,
        or None if the matrices cannot be multiplied.
    """
    # Check if the number of columns in mat1 equals the number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Perform matrix multiplication
    result = [
        [sum(a * b for a, b in zip(row, col))
         for col in zip(*mat2)] for row in mat1
    ]

    return result
