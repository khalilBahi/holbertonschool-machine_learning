#!/usr/bin/env python3
"""2. GRU Cell"""
import numpy as np


class GRUCell:
    def __init__(self, i, h, o):
        """Initialize the GRU cell.

        Args:
            i (int): Dimensionality of the input data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Weights for update gate (Wz), reset gate (Wr), intermediate hidden
        # state (Wh), and output (Wy)
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        # Biases for update gate (bz), reset gate (br), intermediate hidden
        # state (bh), and output (by)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h)
            x_t (numpy.ndarray): Input data of shape (m, i)

        Returns:
            h_next (numpy.ndarray): Next hidden state
            y (numpy.ndarray): Output of the cell
        """
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = self.sigmoid(np.dot(concat, self.Wz) + self.bz)

        # Reset gate
        r_t = self.sigmoid(np.dot(concat, self.Wr) + self.br)

        # Intermediate hidden state
        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
