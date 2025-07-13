#!/usr/bin/env python3
"""
TD(λ) algorithm for value function estimation
"""
import numpy as np


def td_lambtha(
        env,
        V,
        policy,
        lambtha,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99):
    """
    Performs the TD(λ) algorithm for value function estimation

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and
        returns the next action to take
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    # Make a copy of V to avoid modifying the original
    V = V.copy()

    for _ in range(episodes):
        # Initialize eligibility traces for this episode
        eligibility_traces = np.zeros_like(V)

        # Reset environment and get initial state
        state, _ = env.reset()

        # Generate episode trajectory
        for _ in range(max_steps):
            # Get action from policy
            action = policy(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Calculate TD error
            td_error = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace for current state
            eligibility_traces[state] += 1

            # Update all states based on their eligibility traces
            V += alpha * td_error * eligibility_traces

            # Decay eligibility traces
            eligibility_traces *= gamma * lambtha

            # Check if episode is done
            if terminated or truncated:
                break

            state = next_state

    return V
