#!/usr/bin/env python3
"""
This module provides a function to perform element-wise addition of two arrays.

The `add_arrays` function takes two arrays (lists of numbers) and returns a
new array where each element is the sum of the corresponding elements in the
input arrays. If the arrays have different lengths, the function returns None.

Example:
    >>> add_arrays([1, 2, 3], [4, 5, 6])
    [5, 7, 9]
    >>> add_arrays([1, 2], [3, 4, 5])
    None
"""


def add_arrays(arr1, arr2):
    """
    Perform element-wise addition of two arrays.

    The function takes two arrays and returns a new array where each element
    is the sum of the corresponding elements
    from the input arrays. If the arrays
    have different lengths, it returns None.

    Args:
        arr1 (list of int/float): The first array.
        arr2 (list of int/float): The second array.

    Returns:
        list of int/float: A new array with
        the element-wise sums of the inputs,
        or None if the arrays have different lengths.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
