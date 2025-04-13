#!/usr/bin/env python3
"""Task 0. Markov Chain"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Args:
        Observation: numpy.ndarray of shape (T,) - indices of observations
        Emission: numpy.ndarray of shape (N, M) - emission probabilities
        Transition: numpy.ndarray of shape (N, N) - transition probabilities
        Initial: numpy.ndarray of shape (N, 1) - initial state probabilities

    Returns:
        tuple: (P, F) where
            P: float - likelihood of observations given the model
            F: numpy.ndarray of shape (N, T) - forward path probabilities
            or (None, None) on failure
    """
    # Get dimensions
    T = Observation.shape[0]  # Number of observations
    N, M = Emission.shape     # Number of states

    # Initialize forward probability matrix
    F = np.zeros((N, T))

    # Compute initial forward probabilities
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Compute forward probabilities for each time step
    for t in range(1, T):
        for j in range(N):
            # Sum of previous probabilities times transition probs, times
            # emission prob
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]) * \
                Emission[j, Observation[t]]

    # Compute total likelihood
    P = np.sum(F[:, T - 1])

    return P, F
