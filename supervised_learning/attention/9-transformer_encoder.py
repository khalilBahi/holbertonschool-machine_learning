#!/usr/bin/env python3
"""8. Transformer Encoder"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encoder class that inherits from tensorflow.keras.layers.Layer"""

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_len,
            drop_rate=0.1):
        """
        Initialize the Encoder

        Args:
            N (int): Number of blocks in the encoder
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in the fully connected layer
            input_vocab (int): Size of the input vocabulary
            max_seq_len (int): Maximum sequence length possible
            drop_rate (float): Dropout rate
        """
        super(Encoder, self).__init__()

        # Set public instance attributes
        self.N = N
        self.dm = dm

        # Create embedding layer for inputs
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)

        # Create positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Create list of encoder blocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        # Create dropout layer for positional encodings
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=None, mask=None):
        """
        Forward pass through the encoder

        Args:
            x: tensor of shape (batch, input_seq_len)
                containing input to the encoder
            training: boolean to determine if the model is training
            mask: mask to be applied for multi head attention

        Returns:
            tensor of shape (batch, input_seq_len, dm)
                containing the encoder output
        """
        # Get sequence length
        seq_len = tf.shape(x)[1]

        # Apply embedding and scale by sqrt(dm)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        # Only use the positional encodings up to the sequence length
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout to the sum of embeddings and positional encodings
        x = self.dropout(x, training=training)

        # Pass through all encoder blocks
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x
