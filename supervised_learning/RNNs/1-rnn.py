#!/usr/bin/env python3
"""1. RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: Instance of RNNCell for forward propagation
        X: Input data, numpy.ndarray of shape (t, m, i)
            t: Maximum number of time steps
            m: Batch size
            i: Dimensionality of the data
        h_0: Initial hidden state, numpy.ndarray of shape (m, h)
            h: Dimensionality of the hidden state

    Returns:
        H: numpy.ndarray of shape (t+1, m, h) containing all hidden states
        Y: numpy.ndarray of shape (t, m, o) containing all outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    # Initialize arrays to store hidden states and outputs
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden state
    H[0] = h_0

    # Iterate through each time step
    for step in range(t):
        # Perform forward propagation for one time step
        h_next, y = rnn_cell.forward(H[step], X[step])
        # Store the next hidden state and output
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
