#!/usr/bin/env python3
"""Task 13: 14. Batch Normalization Upgraded"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tf.Tensor): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation (function or None): The activation
        function to apply on the output.

    Returns:
        tf.Tensor: The output of the batch normalization layer.
    """
    # Initialize the Dense layer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, kernel_initializer=init, use_bias=False)
    z = layer(prev)

    # Initialize gamma and beta as vectors of shape (1, n)
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)

    # Compute mean and variance along the batch axis
    mean, variance = tf.nn.moments(z, axes=[0], keepdims=True)

    # Normalize, scale, and shift
    epsilon = 1e-7
    normalized = tf.nn.batch_normalization(
        z, mean, variance, beta, gamma, epsilon)

    # Apply activation function
    return activation(normalized)
