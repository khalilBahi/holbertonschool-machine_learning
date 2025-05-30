#!/usr/bin/env python3
"""0. RNN Encoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder class that inherits from tensorflow.keras.layers.Layer"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN Encoder

        Args:
            vocab (int): Size of the input vocabulary
            embedding (int): Dimensionality of the embedding vector
            units (int): Number of hidden units in the RNN cell
            batch (int): Batch size
        """
        super(RNNEncoder, self).__init__()

        # Set public instance attributes
        self.batch = batch
        self.units = units

        # Create embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        # Create GRU layer with glorot_uniform recurrent initializer
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initialize the hidden states for the RNN cell to a tensor of zeros

        Returns:
            tensor: A tensor of shape (batch, units) containing the
                initialized hidden states
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass through the encoder

        Args:
            x (tensor): Input tensor of shape (batch, input_seq_len)
                containing word indices
            initial (tensor): Initial hidden state of shape (batch, units)

        Returns:
            outputs (tensor): Outputs of shape (batch, input_seq_len, units)
            hidden (tensor): Last hidden state of shape (batch, units)
        """
        # Pass input through embedding layer
        x = self.embedding(x)

        # Pass through GRU layer
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
