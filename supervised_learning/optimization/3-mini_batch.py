#!/usr/bin/env python3
"""Task 3: 3. Mini-Batch"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Create mini-batches from input data.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx).
        Y (numpy.ndarray): Labels of shape (m, ny).
        batch_size (int): Number of data points in a mini-batch.

    Returns:
        list: A list of mini-batches, where each
        mini-batch is a tuple (X_batch, Y_batch).
    """
    # First shuffle the data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X_shuffled.shape[0]  # Number of data points
    mini_batches = []

    # Create mini-batches
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
