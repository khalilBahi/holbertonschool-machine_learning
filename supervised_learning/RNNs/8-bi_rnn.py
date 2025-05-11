#!/usr/bin/env python3
"""8. Bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN.

    Args:
        bi_cell: Instance of BidirectionalCell.
        X: Input data, numpy.ndarray of shape (t, m, i).
            t: Number of time steps.
            m: Batch size.
            i: Dimensionality of the data.
        h_0: Initial forward hidden state, numpy.ndarray of shape (m, h).
            h: Dimensionality of the hidden state.
        h_t: Initial backward hidden state, numpy.ndarray of shape (m, h).

    Returns:
        H: numpy.ndarray of shape (t, m, 2*h) containing
        concatenated hidden states.
        Y: numpy.ndarray of shape (t, m, o) containing outputs.
    """
    # Extract dimensions
    t, m, i = X.shape
    h = h_0.shape[1]
    o = bi_cell.Wy.shape[1]

    # Initialize arrays for hidden states
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))
    H = np.zeros((t, m, 2 * h))

    # Forward direction: time 0 to t-1
    h_prev = h_0
    for step in range(t):
        h_next = bi_cell.forward(h_prev, X[step])
        H_forward[step] = h_next
        h_prev = h_next

    # Backward direction: time t-1 to 0
    h_next = h_t
    for step in range(t - 1, -1, -1):
        h_prev = bi_cell.backward(h_next, X[step])
        H_backward[step] = h_prev
        h_next = h_prev

    # Concatenate forward and backward hidden states
    for step in range(t):
        H[step, :, :h] = H_forward[step]
        H[step, :, h:] = H_backward[step]

    # Compute outputs
    Y = bi_cell.output(H)

    return H, Y
