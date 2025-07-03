#!/usr/bin/env python3
"""
4. Play
"""

import numpy as np
import re


def convert_ansi_to_backticks(text):
    """
    Convert ANSI color codes to backticks for agent position
    """
    # Replace ANSI highlight codes with backticks
    # \x1b[41m starts red background, \x1b[0m ends formatting
    text = re.sub(r'\x1b\[41m(.)\x1b\[0m', r'`\1`', text)
    return text


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode

    Args:
        env: The FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: Maximum number of steps in the episode

    Returns:
        total_rewards: The total rewards for the episode
        rendered_outputs: List of rendered outputs
        representing board state at each step
    """
    # Reset environment and get initial state
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    # Render initial state
    initial_render = convert_ansi_to_backticks(env.render())
    rendered_outputs.append(initial_render)

    for step in range(max_steps):
        # Always exploit - choose action with highest Q-value
        action = np.argmax(Q[state])

        # Take action
        new_state, reward, done, truncated, _ = env.step(action)

        # Update total rewards
        total_rewards += reward

        # Render the state after taking action
        rendered_output = convert_ansi_to_backticks(env.render())
        rendered_outputs.append(rendered_output)

        # Move to next state
        state = new_state

        # Check if episode is done
        if done or truncated:
            break

    return total_rewards, rendered_outputs
