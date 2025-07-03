#!/usr/bin/env python3
"""
0. Load the Environment
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium

    Args:
        desc: Either None or a list of lists containing a custom description
              of the map to load for the environment
        map_name: Either None or a string containing the pre-made map to load
        is_slippery: A boolean to determine if the ice is slippery

    Note: If both desc and map_name are None, the environment will load
          a randomly generated 8x8 map

    Returns:
        The FrozenLake environment
    """
    # Create the environment with the specified parameters
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )

    return env
