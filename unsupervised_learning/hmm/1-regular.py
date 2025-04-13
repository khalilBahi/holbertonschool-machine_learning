#!/usr/bin/env python3
"""Task 0. Markov Chain"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) - transition matrix

    Returns:
        numpy.ndarray of shape (1, n) with steady state probabilities,
        or None if P is not a regular Markov chain or on failure
    """
    n = P.shape[0]

    # Check if P^k > 0 for some reasonable k
    P_k = P.copy()
    for i in range(n * n):
        if np.all(P_k > 0):
            break
        P_k = np.dot(P_k, P)
    else:
        # If no power has all positive entries, it's not regular
        return None
    try:

        # Create A = P^T - I
        A = P.T - np.eye(n)

        # Replace last row with ones for normalization condition (sum = 1)
        A[-1] = np.ones(n)

        # Create b vector (zeros with 1 in last position)
        b = np.zeros(n)
        b[-1] = 1

        # Solve the system Ax = b
        pi = np.linalg.solve(A, b)

        # Verify solution
        if not np.allclose(
                np.dot(pi, P), pi) or not np.allclose(np.sum(pi), 1):
            return None

        if np.any(pi < 0):  # Probabilities can't be negative
            return None

        # Return as (1, n) array
        return pi.reshape(1, n)

    except Exception:
        return None
