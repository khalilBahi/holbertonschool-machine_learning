#!/usr/bin/env python3
"""4. Scaled Dot Product Attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate the scaled dot product attention

    Args:
        Q: tensor with last two dimensions (..., seq_len_q, dk)
            containing query matrix
        K: tensor with last two dimensions (..., seq_len_v, dk)
            containing key matrix
        V: tensor with last two dimensions (..., seq_len_v, dv)
            containing value matrix
        mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
            containing optional mask

    Returns:
        output: tensor with last two dimensions (..., seq_len_q, dv)
            containing scaled dot product attention
        weights: tensor with last two dimensions (..., seq_len_q, seq_len_v)
            containing attention weights
    """
    # Get the dimension of the key vectors (dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Calculate Q * K^T
    # Q shape: (..., seq_len_q, dk)
    # K shape: (..., seq_len_v, dk)
    # matmul result shape: (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale by sqrt(dk)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided
    if mask is not None:
        # Add large negative value to masked positions
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax to get attention weights
    # Shape: (..., seq_len_q, seq_len_v)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Apply attention weights to values
    # attention_weights shape: (..., seq_len_q, seq_len_v)
    # V shape: (..., seq_len_v, dv)
    # output shape: (..., seq_len_q, dv)
    output = tf.matmul(attention_weights, V)

    return output, attention_weights
