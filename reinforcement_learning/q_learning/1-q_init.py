#!/usr/bin/env python3
"""
1. Initialize Q-table
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table for a FrozenLake environment

    Args:
        env: The FrozenLakeEnv instance

    Returns:
        The Q-table as a numpy.ndarray of zeros with shape (states, actions)
    """
    # Get the number of states from the observation space
    num_states = env.observation_space.n

    # Get the number of actions from the action space
    num_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q = np.zeros((num_states, num_actions))

    return Q
