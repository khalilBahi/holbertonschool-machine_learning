#!/usr/bin/env python3
"""Task 6: 6. Create a Layer with Dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Parameters:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function for the new layer
    keep_prob -- probability that a node will be kept (1 - dropout rate)
    training -- boolean indicating whether the model is in training mode

    Returns:
    output -- the output tensor of the new layer
    """
    # Initialize weights using VarianceScaling with fan_avg mode
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg")

    # Create a Dense layer with the specified activation and custom initializer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)

    # Apply dropout with rate = 1 - keep_prob
    if training:
        layer = tf.nn.dropout(layer(prev), rate=1 - keep_prob)

    return layer
