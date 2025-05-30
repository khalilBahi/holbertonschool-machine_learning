#!/usr/bin/env python3
"""9. Transformer Decoder"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder class that inherits from tensorflow.keras.layers.Layer"""

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_len,
            drop_rate=0.1):
        """
        Initialize the Decoder

        Args:
            N (int): Number of blocks in the decoder
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in the fully connected layer
            target_vocab (int): Size of the target vocabulary
            max_seq_len (int): Maximum sequence length possible
            drop_rate (float): Dropout rate
        """
        super(Decoder, self).__init__()

        # Set public instance attributes
        self.N = N
        self.dm = dm

        # Create embedding layer for targets
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)

        # Create positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Create list of decoder blocks
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        # Create dropout layer for positional encodings
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(
            self,
            x,
            encoder_output,
            training=None,
            look_ahead_mask=None,
            padding_mask=None):
        """
        Forward pass through the decoder

        Args:
            x: tensor of shape (batch, target_seq_len)
                containing input to the decoder
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                containing encoder output
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to the first
                multi head attention layer
            padding_mask: mask to be applied to the second
                multi head attention layer

        Returns:
            tensor of shape (batch, target_seq_len, dm)
                containing the decoder output
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

        # Pass through all decoder blocks
        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask)

        return x
