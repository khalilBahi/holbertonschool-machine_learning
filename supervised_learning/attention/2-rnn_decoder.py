#!/usr/bin/env python3
"""2. RNN Decoder"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder class that inherits from tensorflow.keras.layers.Layer"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN Decoder

        Args:
            vocab (int): Size of the output vocabulary
            embedding (int): Dimensionality of the embedding vector
            units (int): Number of hidden units in the RNN cell
            batch (int): Batch size
        """
        super(RNNDecoder, self).__init__()

        # Create embedding layer for target vocabulary
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        # Create GRU layer with glorot_uniform recurrent initializer
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        # Dense layer for output projection to vocabulary size
        self.F = tf.keras.layers.Dense(vocab)

        # Create attention layer
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Decode one step for machine translation

        Args:
            x (tensor): Previous word in target sequence of shape (batch, 1)
            s_prev (tensor): Previous decoder hidden state of shape
                (batch, units)
            hidden_states (tensor): Encoder outputs of shape
                (batch, input_seq_len, units)

        Returns:
            y (tensor): Output word as logits of shape (batch, vocab)
            s (tensor): New decoder hidden state of shape (batch, units)
        """
        # Use attention to get context vector
        # context shape: (batch, units)
        context, _ = self.attention(s_prev, hidden_states)

        # Embed the input word
        # x shape: (batch, 1) -> (batch, 1, embedding)
        x_embedded = self.embedding(x)

        # Expand context to match sequence dimension
        # context shape: (batch, units) -> (batch, 1, units)
        context_expanded = tf.expand_dims(context, 1)

        # Concatenate context vector with embedded input
        # (context first, then x)
        # Shape: (batch, 1, units + embedding)
        concatenated = tf.concat([context_expanded, x_embedded], axis=-1)

        # Pass through GRU layer
        # outputs shape: (batch, 1, units)
        # s shape: (batch, units)
        outputs, s = self.gru(concatenated, initial_state=s_prev)

        # Remove sequence dimension from outputs
        # outputs shape: (batch, 1, units) -> (batch, units)
        outputs = tf.squeeze(outputs, axis=1)

        # Apply Dense layer to get output logits
        # y shape: (batch, vocab)
        y = self.F(outputs)

        return y, s
