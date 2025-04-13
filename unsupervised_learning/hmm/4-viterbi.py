#!/usr/bin/env python3
"""Task 0. Markov Chain"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden
    states for a hidden Markov model.

    Args:
        Observation: numpy.ndarray of shape (T,) - indices of observations
        Emission: numpy.ndarray of shape (N, M) - emission probabilities
        Transition: numpy.ndarray of shape (N, N) - transition probabilities
        Initial: numpy.ndarray of shape (N, 1) - initial state probabilities

    Returns:
        tuple: (path, P) where
            path: list of length T - most likely sequence of hidden states
            P: float - probability of the path sequence
            or (None, None) on failure
    """
    # Get dimensions
    T = Observation.shape[0]  # Number of observations
    N = Emission.shape     # Number of states, number of possible observations

    # Initialize Viterbi probability matrix and backpointer
    V = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Compute initial Viterbi probabilities
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Compute Viterbi probabilities for each time step
    for t in range(1, T):
        for j in range(N):
            # Calculate probabilities for all transitions to state j
            prob = V[:, t - 1] * Transition[:, j]
            # Store max probability and corresponding previous state
            V[j, t] = np.max(prob) * Emission[j, Observation[t]]
            backpointer[j, t] = np.argmax(prob)

    # Initialize path with most likely final state
    path = [0] * T
    path[T - 1] = np.argmax(V[:, T - 1])

    # Backtrack to find most likely path
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    # Compute probability of the path
    P = np.max(V[:, T - 1])

    return path, P
