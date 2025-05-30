#!/usr/bin/env python3
"""1. Self Attention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """SelfAttention class that inherits from tensorflow.keras.layers.Layer"""

    def __init__(self, units):
        """
        Initialize the Self Attention layer

        Args:
            units (int): Number of hidden units in the alignment model
        """
        super(SelfAttention, self).__init__()

        # Dense layer to be applied to the previous decoder hidden state
        self.W = tf.keras.layers.Dense(units)

        # Dense layer to be applied to the encoder hidden states
        self.U = tf.keras.layers.Dense(units)

        # Dense layer to be applied to the tanh of the sum of W and U outputs
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Calculate the attention for machine translation

        Args:
            s_prev (tensor): Previous decoder hidden state of shape
                (batch, units)
            hidden_states (tensor): Encoder outputs of shape
                (batch, input_seq_len, units)

        Returns:
            context (tensor): Context vector of shape (batch, units)
            weights (tensor): Attention weights of shape
                (batch, input_seq_len, 1)
        """
        # Expand s_prev to match the sequence length dimension
        # s_prev shape: (batch, units) -> (batch, 1, units)
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Apply W to the previous decoder hidden state
        # Shape: (batch, 1, units)
        W_s = self.W(s_prev_expanded)

        # Apply U to the encoder hidden states
        # Shape: (batch, input_seq_len, units)
        U_h = self.U(hidden_states)

        # Add W_s and U_h (broadcasting will handle the dimensions)
        # Shape: (batch, input_seq_len, units)
        combined = W_s + U_h

        # Apply tanh activation
        tanh_combined = tf.nn.tanh(combined)

        # Apply V to get attention scores
        # Shape: (batch, input_seq_len, 1)
        scores = self.V(tanh_combined)

        # Apply softmax to get attention weights
        # Shape: (batch, input_seq_len, 1)
        weights = tf.nn.softmax(scores, axis=1)

        # Calculate context vector as weighted sum of encoder hidden states
        # weights shape: (batch, input_seq_len, 1)
        # hidden_states shape: (batch, input_seq_len, units)
        # context shape: (batch, units)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
