#!/usr/bin/env python3
"""4. Deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    This method performs forward propagation for a deep RNN
    rnn_cells is a list of RNNCell instances of length num_layers that will
    be used for the forward propagation
    num_layers is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state, given as a
    numpy.ndarray of shape (num_layers, m, h)
    h is the dimensionality of the hidden state
    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """
    # Extract dimensions: num_layers (layers), m (batch size),
    # h (hidden state dim) from h_0
    # Note: This assumes h_0 has shape (num_layers, m, h), but i is not used
    # here
    h_next = h_0
    num_layers, m, i = h_0.shape

    # Initialize H to store all hidden states
    H = np.zeros((X.shape[0] + 1, num_layers, m, i))

    # Set initial hidden states in H[0]
    H[0, :, :, :] = h_0

    # Initialize list to collect outputs Y
    Y = []

    # Iterate over each time step (t time steps from X.shape[0])
    for t in range(X.shape[0]):
        # Start with input for the first layer at time step t
        a_prev = X[t]

        # Process each layer
        for layer_idx in range(num_layers):
            # Perform forward propagation for the current layer
            h_next, y = rnn_cells[layer_idx].forward(H[t, layer_idx], a_prev)

            # Set the input for the next layer to h_next (hidden state)
            a_prev = h_next

            # Store the new hidden state in H for the next time step
            H[t + 1, layer_idx, :, :] = h_next

        # Collect the output from the final layer (y from the last layer)
        Y.append(y)

    return H, np.array(Y)
