#!/usr/bin/env python3
"""
1. Minor
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose determinant should be calculated

    returns:
        the determinant of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    if height is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) is 0 and height is 1:
            return 1
        if len(row) != height:
            raise ValueError("matrix must be a square matrix")
    if height is 1:
        return matrix[0][0]
    if height is 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return ((a * d) - (b * c))
    multiplier = 1
    d = 0
    for i in range(height):
        element = matrix[0][i]
        sub_matrix = []
        for row in range(height):
            if row == 0:
                continue
            new_row = []
            for column in range(height):
                if column == i:
                    continue
                new_row.append(matrix[row][column])
            sub_matrix.append(new_row)
        d += (element * multiplier * determinant(sub_matrix))
        multiplier *= -1
    return (d)


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix

    parameters:
        matrix [list of lists]:
            matrix whose minor matrix should be calculated

    returns:
        the minor matrix of matrix

    raises:
        TypeError: if matrix is not a list of lists (or is empty)
        ValueError: if matrix is not a non-empty square matrix
    """
    # Validate type and non-emptiness
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        # Match project convention: empty -> TypeError with this message
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Handle 1x1 (and degenerate [ [] ] case) explicitly
    if n == 1:
        if len(matrix[0]) == 0:
            return [[1]]
        if len(matrix[0]) != 1:
            raise ValueError("matrix must be a non-empty square matrix")
        return [[1]]

    # Validate square shape for n > 1
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # Build the minor matrix
    minors = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Construct submatrix excluding row i and column j
            sub = []
            for r in range(n):
                if r == i:
                    continue
                new_row = []
                for c in range(n):
                    if c == j:
                        continue
                    new_row.append(matrix[r][c])
                sub.append(new_row)
            minor_val = determinant(sub)
            minor_row.append(minor_val)
        minors.append(minor_row)
    return minors