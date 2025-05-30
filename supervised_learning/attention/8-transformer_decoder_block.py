#!/usr/bin/env python3
"""7. Transformer Decoder Block"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class that inherits from tensorflow.keras.layers.Layer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the Decoder Block

        Args:
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in the fully connected layer
            drop_rate (float): Dropout rate
        """
        super(DecoderBlock, self).__init__()

        # Multi-head attention layers
        self.mha1 = MultiHeadAttention(dm, h)  # Masked self-attention
        self.mha2 = MultiHeadAttention(dm, h)  # Cross-attention with encoder

        # Feed-forward network layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(
            self,
            x,
            encoder_output,
            training=None,
            look_ahead_mask=None,
            padding_mask=None):
        """
        Forward pass through the decoder block

        Args:
            x: tensor of shape (batch, target_seq_len, dm)
                containing input to decoder block
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                containing encoder output
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to the first
                multi head attention layer
            padding_mask: mask to be applied to the second
                multi head attention layer

        Returns:
            tensor of shape (batch, target_seq_len, dm)
                containing the block's output
        """
        # Block 1: Masked multi-head self-attention
        # Q, K, V all come from the decoder input (x)
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)

        # Apply dropout to first attention output
        attn1 = self.dropout1(attn1, training=training)

        # Add & Norm (residual connection + layer normalization)
        out1 = self.layernorm1(x + attn1)

        # Block 2: Multi-head cross-attention with encoder output
        # Q comes from decoder (out1), K and V come from encoder output
        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)

        # Apply dropout to second attention output
        attn2 = self.dropout2(attn2, training=training)

        # Add & Norm (residual connection + layer normalization)
        out2 = self.layernorm2(out1 + attn2)

        # Block 3: Feed-forward network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)

        # Apply dropout to feed-forward output
        ffn_output = self.dropout3(ffn_output, training=training)

        # Add & Norm (residual connection + layer normalization)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
