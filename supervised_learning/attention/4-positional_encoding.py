#!/usr/bin/env python3
"""3. Positional Encoding"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate the positional encoding for a transformer

    Args:
        max_seq_len (int): Maximum sequence length
        dm (int): Model depth

    Returns:
        numpy.ndarray: Positional encoding vectors of shape (max_seq_len, dm)
    """
    # Initialize the positional encoding matrix
    PE = np.zeros((max_seq_len, dm))

    # Create position indices (0, 1, 2, ..., max_seq_len-1)
    position = np.arange(max_seq_len)[:, np.newaxis]  # Shape: (max_seq_len, 1)

    # Create dimension indices (0, 1, 2, ..., dm-1)
    # We need to calculate for every pair of dimensions (2i, 2i+1)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Apply sine to even indices
    PE[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
