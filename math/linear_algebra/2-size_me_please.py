#!/usr/bin/env python3

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
