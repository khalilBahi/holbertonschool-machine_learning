#!/usr/bin/env python3
"""
This module provides a function for element-wise operations on NumPy arrays.

The `np_elementwise` function performs element-wise addition, subtraction,
multiplication, and division, returning the results as a tuple.
"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division
    on two NumPy arrays.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray or scalar): The second
        array or scalar for operations.

    Returns:
        tuple: A tuple containing four NumPy arrays corresponding to the
               element-wise sum, difference, product, and quotient.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
