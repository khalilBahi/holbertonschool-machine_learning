#!/usr/bin/env python3
"""
2. Epsilon Greedy
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Args:
        Q: numpy.ndarray containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation

    Returns:
        The next action index
    """
    # Sample a random probability to decide explore vs exploit
    p = np.random.uniform(0, 1)

    # If p < epsilon, explore (choose random action)
    if p < epsilon:
        # Pick random action from all possible actions
        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploit: choose action with highest Q-value for current state
        action = np.argmax(Q[state])

    return action
