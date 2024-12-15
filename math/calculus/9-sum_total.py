#!/usr/bin/env python3
"""task 9: 9. Our life is the sum total of all the decisions
we make every day, and those decisions are determined by our priorities"""

import numpy as np


def summation_i_squared(n):
    """
    Calculate the sum of squares of integers from 1 to n.

    Args:
        n (int): The stopping condition.

    Returns:
        int: The sum of squares from 1 to n if n is valid.
        None: If n is not a valid positive integer.
    """
    if not isinstance(n, int) or n < 1:
        return None

    Sum = sum(np.square(np.arange(1, n + 1)))
    return Sum
