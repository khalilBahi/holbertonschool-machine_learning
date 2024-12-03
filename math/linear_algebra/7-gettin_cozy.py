#!/usr/bin/env python3
"""
This module provides a function to concatenate
two 2D matrices along a specific axis.

The `cat_matrices2D` function allows for the concatenation of two 2D matrices
along rows (axis=0) or columns (axis=1). If the matrices cannot be concatenated
due to incompatible dimensions or an invalid axis, the function returns None.

Example:
    >>> cat_matrices2D([[1, 2], [3, 4]], [[5, 6]], axis=0)
    [[1, 2], [3, 4], [5, 6]]
    >>> cat_matrices2D([[1, 2], [3, 4]], [[5], [6]], axis=1)
    [[1, 2, 5], [3, 4, 6]]
    >>> cat_matrices2D([[1, 2]], [[3], [4]], axis=1)
    None
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along a specific axis.

    This function concatenates two 2D matrices
    along the specified axis (rows or columns).
    If the matrices have incompatible dimensions
    or the axis is invalid, it returns None.

    Args:
        mat1 (list of lists of int/float): The first 2D matrix.
        mat2 (list of lists of int/float): The second 2D matrix.
        axis (int, optional): The axis along which to concatenate.
                              0 for rows (default), 1 for columns.

    Returns:
        list of lists of int/float: A new 2D
        matrix resulting from the concatenation,
        or None if the matrices cannot be concatenated.

    """
    # Concatenation along rows (axis=0)
    if axis == 0:
        # Check if all rows have the same number of columns
        if all(len(row) == len(mat1[0]) for row in mat2):
            return [row[:] for row in mat1] + [row[:] for row in mat2]
        else:
            return None

    # Concatenation along columns (axis=1)
    elif axis == 1:
        # Check if both matrices have the same number of rows
        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        else:
            return None

    # If the axis is invalid
    return None
