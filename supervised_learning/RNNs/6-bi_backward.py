#!/usr/bin/env python3
"""6. Bidirectional Cell Backward"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional RNN cell."""

    def __init__(self, i, h, o):
        """
        Initialize the bidirectional cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.

        Attributes:
            Whf (numpy.ndarray): Weight matrix for forward
            hidden state, shape (h+i, h).
            Whb (numpy.ndarray): Weight matrix for
            backward hidden state, shape (h+i, h).
            Wy (numpy.ndarray): Weight matrix for output, shape (2*h, o).
            bhf (numpy.ndarray): Bias for forward hidden state, shape (1, h).
            bhb (numpy.ndarray): Bias for backward hidden state, shape (1, h).
            by (numpy.ndarray): Bias for output, shape (1, o).
        """
        # Initialize weights with random normal distribution
        self.Whf = np.random.randn(h + i, h)
        self.Whb = np.random.randn(h + i, h)
        self.Wy = np.random.randn(2 * h, o)

        # Initialize biases as zeros
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculate the forward hidden state for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state, shape (m, h).
            x_t (numpy.ndarray): Input data, shape (m, i).

        Returns:
            h_next (numpy.ndarray): Next hidden state
            in forward direction, shape (m, h).
        """
        # Concatenate h_prev and x_t along axis 1
        # Shape: (m, h+i), e.g., (8, 15+10) = (8, 25)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute forward hidden state: h_next = tanh(concat @ Whf + bhf)
        # concat: (m, h+i), Whf: (h+i, h) -> (m, h)
        # bhf: (1, h) broadcasts to (m, h)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculate the backward hidden state for one time step.

        Args:
            h_next (numpy.ndarray): Next hidden state
            in backward direction, shape (m, h).
            x_t (numpy.ndarray): Input data, shape (m, i).

        Returns:
            h_prev (numpy.ndarray): Previous hidden state
            in backward direction, shape (m, h).
        """
        # Concatenate h_next and x_t along axis 1
        # Shape: (m, h+i), e.g., (8, 15+10) = (8, 25)
        concat = np.concatenate((h_next, x_t), axis=1)

        # Compute backward hidden state: h_prev = tanh(concat @ Whb + bhb)
        # concat: (m, h+i), Whb: (h+i, h) -> (m, h)
        # bhb: (1, h) broadcasts to (m, h)
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev
