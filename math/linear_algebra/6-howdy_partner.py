#!/usr/bin/env python3
"""
This module provides a function to concatenate two arrays.

The `cat_arrays` function takes two arrays (lists of numbers) and returns 
a new array that is the concatenation of the input arrays.

Example:
    >>> cat_arrays([1, 2, 3], [4, 5, 6])
    [1, 2, 3, 4, 5, 6]
    >>> cat_arrays([], [7, 8, 9])
    [7, 8, 9]
"""

def cat_arrays(arr1, arr2):
    """
    Concatenate two arrays.

    This function returns a new array that contains all the elements 
    from the first array followed by all the elements from the second array.

    Args:
        arr1 (list of int/float): The first array.
        arr2 (list of int/float): The second array.

    Returns:
        list of int/float: A new array that is the concatenation of the two input arrays.

    """
    return arr1 + arr2
