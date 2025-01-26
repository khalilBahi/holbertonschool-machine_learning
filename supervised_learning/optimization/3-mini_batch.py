#!/usr/bin/env python3
"""Task 3: 3. Mini-Batch"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(x_shuffled, y_shuffled, batch_size):
    """
    Create mini-batches from already shuffled data.

    Args:
        x_shuffled (numpy.ndarray): Shuffled input data of shape (m, nx).
        y_shuffled (numpy.ndarray): Shuffled labels of shape (m, ny).
        batch_size (int): Number of data points in a mini-batch.

    Returns:
        list: A list of mini-batches, where each mini-batch is a tuple (X_batch, Y_batch).
    """
    m = x_shuffled.shape[0]  # Number of data points
    mini_batches = []

    # If batch_size > m, set batch_size = m
    if batch_size > m:
        batch_size = m

    # Create mini-batches
    for i in range(0, m, batch_size):
        X_batch = x_shuffled[i:i + batch_size]
        Y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches