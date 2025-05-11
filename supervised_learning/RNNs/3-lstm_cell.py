#!/usr/bin/env python3
"""3. LSTM Cell"""
import numpy as np


class LSTMCell:
    """class LSTMCell"""
    def __init__(self, i, h, o):
        """Initialize the LSTM cell.

        Args:
            i (int): Dimensionality of the input data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Weights for forget gate (Wf), update gate (Wu), cell state candidate
        # (Wc), output gate (Wo), and output (Wy)
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        # Biases for forget gate (bf), update gate (bu), cell state candidate
        # (bc), output gate (bo), and output (by)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h)
            c_prev (numpy.ndarray): Previous cell state of shape (m, h)
            x_t (numpy.ndarray): Input data of shape (m, i)

        Returns:
            h_next (numpy.ndarray): Next hidden state
            c_next (numpy.ndarray): Next cell state
            y (numpy.ndarray): Output of the cell
        """
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = self.sigmoid(np.dot(concat, self.Wf) + self.bf)

        # Update gate
        u_t = self.sigmoid(np.dot(concat, self.Wu) + self.bu)

        # Cell state candidate
        c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)

        # Next cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate
        o_t = self.sigmoid(np.dot(concat, self.Wo) + self.bo)

        # Next hidden state
        h_next = o_t * np.tanh(c_next)

        # Output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
