#!/usr/bin/env python3

import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """Initialize the RNN cell.

        Args:
            i (int): Dimensionality of the input data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Weight matrix for concatenated [h_prev, x_t]
        self.Wh = np.random.randn(i + h, h)
        # Weight matrix for output
        self.Wy = np.random.randn(h, o)
        # Bias for hidden state
        self.bh = np.zeros((1, h))
        # Bias for output
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h)
            x_t (numpy.ndarray): Input data of shape (m, i)

        Returns:
            h_next (numpy.ndarray): Next hidden state
            y (numpy.ndarray): Output of the cell
        """

        # Compute next hidden state with tanh activation
        h_next = np.tanh(np.dot(np.concatenate(
            [h_prev, x_t], axis=1), self.Wh) + self.bh)

        # Compute output logits
        logits = np.dot(h_next, self.Wy) + self.by

        # Apply softmax activation for output
        exp_logits = np.exp(logits - np.max(logits))
        y = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return h_next, y
