#!/usr/bin/env python3
"""
4. Play
"""

import numpy as np


def convert_ansi_to_backticks(text):
    """
    Convert ANSI color codes to backticks for agent position
    """
    # Replace ANSI highlight codes with backticks
    # \x1b[41m starts red background, \x1b[0m ends formatting
    # Find and replace the ANSI codes manually
    start_code = '\x1b[41m'
    end_code = '\x1b[0m'

    while start_code in text and end_code in text:
        start_idx = text.find(start_code)
        if start_idx == -1:
            break

        # Find the character after the start code
        char_idx = start_idx + len(start_code)
        if char_idx >= len(text):
            break

        # Find the end code after the character
        end_idx = text.find(end_code, char_idx)
        if end_idx == -1:
            break

        # Extract the character between the codes
        char = text[char_idx:end_idx]

        # Replace the entire sequence with backticks around the character
        text = text[:start_idx] + '`' + char + '`' + text[end_idx + len(end_code):]

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

    for _ in range(max_steps):
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
