#!/usr/bin/env python3
"""
Module for Q-learning training algorithm
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning training

    Args:
        env: The FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: Total number of episodes to train over
        max_steps: Maximum number of steps per episode
        alpha: Learning rate
        gamma: Discount rate
        epsilon: Initial threshold for epsilon greedy
        min_epsilon: Minimum value that epsilon should decay to
        epsilon_decay: Decay rate for updating epsilon between episodes

    Returns:
        Q: The updated Q-table
        total_rewards: List containing the rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        # Reset environment for new episode
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Choose action using epsilon-greedy
            action = epsilon_greedy(Q, state, epsilon)

            # Take action and observe result
            new_state, reward, done, truncated, _ = env.step(action)

            # Q-learning update rule
            # Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
            if done:
                # Terminal state - no future rewards
                Q[state, action] = Q[state, action] + \
                    alpha * (reward - Q[state, action])
            else:
                # Non-terminal state - include discounted future rewards
                Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * np.max(Q[new_state]) - Q[state, action]
                )

            # Update episode reward - modify reward if agent falls in hole
            if done and reward == 0:
                episode_reward += -1
            else:
                episode_reward += reward

            # Move to next state
            state = new_state

            # Check if episode is done
            if done or truncated:
                break

        # Store episode reward
        total_rewards.append(episode_reward)

        # Decay epsilon after each episode
        if epsilon > min_epsilon:
            epsilon = epsilon - epsilon_decay

    return Q, total_rewards
