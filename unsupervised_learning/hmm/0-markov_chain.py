#!/usr/bin/env python3
"""Task 0. Markov Chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular state
    after a specified number of iterations.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) - transition matrix
        s: numpy.ndarray of shape (1, n) - initial state probability
        t: int - number of iterations

    Returns:
        numpy.ndarray of shape (1, n) with probabilities after t iterations,
        or None on failure
    """
    try:
        # Calculate the state probabilities after t iterations
        # For t=0, return initial state
        if t == 0:
            return s

        # For t iterations, multiply initial state by P raised to power t
        # Using matrix power for efficiency
        P_t = np.linalg.matrix_power(P, t)
        result = np.dot(s, P_t)

        return result

    except Exception:
        return None
