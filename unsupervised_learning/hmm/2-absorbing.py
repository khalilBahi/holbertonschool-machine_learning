#!/usr/bin/env python3
"""Task 0. Markov Chain"""
import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) - standard transition matrix

    Returns:
        bool: True if the Markov chain is absorbing,
        False otherwise or on failure
    """
    n = P.shape[0]

    # Identify absorbing states (states with P[i,i] = 1)
    absorbing_states = np.where(np.diagonal(P) == 1)[0]

    # For each non-absorbing state, check if an absorbing state is reachable
    # Use a transition matrix to track reachability
    for state in range(n):
        if state in absorbing_states:
            continue

        # Perform a breadth-first search to find paths to absorbing states
        visited = set()
        stack = [state]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            # If we reach an absorbing state, this path is valid
            if current in absorbing_states:
                break

            # Add all possible next states (with non-zero probability)
            next_states = np.where(P[current] > 0)[0]
            stack.extend(next_states)
        else:
            # If we exhaust the search without finding an absorbing state
            return False

    return True
