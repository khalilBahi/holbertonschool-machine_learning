#!/usr/bin/env python3
"""Task 0. Markov Chain"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.

    Args:
        Observation: numpy.ndarray of shape (T,) - indices of observations
        Emission: numpy.ndarray of shape (N, M) - emission probabilities
        Transition: numpy.ndarray of shape (N, N) - transition probabilities
        Initial: numpy.ndarray of shape (N, 1) - initial state probabilities

    Returns:
        tuple: (P, B) where
            P: float - likelihood of observations given the model
            B: numpy.ndarray of shape (N, T) - backward path probabilities
            or (None, None) on failure
    """
    # Get dimensions
    T = Observation.shape[0]  # Number of observations
    N, M = Emission.shape     # Number of states

    # Initialize backward probability matrix
    B = np.zeros((N, T))

    # Set final time step probabilities to 1
    B[:, T - 1] = 1

    # Compute backward probabilities for each time step
    for t in range(T - 2, -1, -1):
        for i in range(N):
            # Sum transition probs times emission probs times next backward
            # probs
            B[i, t] = np.sum(Transition[i, :] * Emission[:,
                             Observation[t + 1]] * B[:, t + 1])

    # Compute total likelihood using initial state and first observation
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
