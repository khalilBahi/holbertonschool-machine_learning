#!/usr/bin/env python3
"""6. Transformer Encoder Block"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """EncoderBlock class that inherits from tensorflow.keras.layers.Layer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the Encoder Block

        Args:
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in the fully connected layer
            drop_rate (float): Dropout rate
        """
        super(EncoderBlock, self).__init__()

        # Multi-head attention layer
        self.mha = MultiHeadAttention(dm, h)

        # Feed-forward network layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def __call__(self, x, training=None, mask=None):
        """Override __call__ to handle positional arguments"""
        return super().__call__(x, training=training, mask=mask)

    def call(self, x, training=None, mask=None):
        """
        Forward pass through the encoder block

        Args:
            x: tensor of shape (batch, input_seq_len, dm)
                containing input to encoder block
            training: boolean to determine if the model is training
            mask: mask to be applied for multi head attention

        Returns:
            tensor of shape (batch, input_seq_len, dm)
                containing the block's output
        """
        # Multi-head self-attention
        # For self-attention, Q, K, V are all the same (x)
        attn_output, _ = self.mha(x, x, x, mask)

        # Apply dropout to attention output
        attn_output = self.dropout1(attn_output, training=training)

        # Add & Norm (residual connection + layer normalization)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)

        # Apply dropout to feed-forward output
        ffn_output = self.dropout2(ffn_output, training=training)

        # Add & Norm (residual connection + layer normalization)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
