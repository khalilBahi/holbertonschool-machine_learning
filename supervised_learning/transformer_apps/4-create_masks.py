#!/usr/bin/env python3
"""4. Create Masks"""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) containing input
        target: tf.Tensor of shape (batch_size, seq_len_out) containing target

    Returns:
        encoder_mask: tf.Tensor padding mask of shape
                     (batch_size, 1, 1, seq_len_in) to be applied in encoder
        combined_mask: tf.Tensor of shape
                      (batch_size, 1, seq_len_out, seq_len_out) used in the 1st
                      attention block in the decoder to pad and mask future
                      tokens. Takes maximum between lookahead mask and decoder
                      target padding mask.
        decoder_mask: tf.Tensor padding mask of shape
                     (batch_size, 1, 1, seq_len_in) used in the 2nd attention
                     block in the decoder
    """

    def create_padding_mask(seq):
        """
        Creates a padding mask for a sequence

        Args:
            seq: tf.Tensor of shape (batch_size, seq_len)

        Returns:
            mask: tf.Tensor of shape (batch_size, 1, 1, seq_len)
                 where 1.0 indicates padding positions and 0.0 indicates
                 valid tokens
        """
        # Create mask where padding tokens (0) are marked as 1.0
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # Add extra dimensions for attention heads and broadcasting
        # Shape: (batch_size, 1, 1, seq_len)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(size):
        """
        Creates a lookahead mask to mask future tokens

        Args:
            size: int, sequence length

        Returns:
            mask: tf.Tensor of shape (size, size) where 1.0 indicates positions
                 that should be masked (future tokens)
        """
        # Create upper triangular matrix with 1s above diagonal
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    # Create encoder padding mask
    # Shape: (batch_size, 1, 1, seq_len_in)
    encoder_mask = create_padding_mask(inputs)

    # Create decoder padding mask (same as encoder mask, used in 2nd attention)
    # Shape: (batch_size, 1, 1, seq_len_in)
    decoder_mask = create_padding_mask(inputs)

    # Create target padding mask
    # Shape: (batch_size, 1, 1, seq_len_out)
    target_padding_mask = create_padding_mask(target)

    # Create lookahead mask for target sequence
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)

    # Combine lookahead mask and target padding mask
    # Take maximum to ensure both future tokens and padding are masked
    # Shape: (batch_size, 1, seq_len_out, seq_len_out)
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
