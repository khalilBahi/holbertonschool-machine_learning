#!/usr/bin/env python3
"""Task 6: 6. Create a Layer with Dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Parameters:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function that should be used on the layer
    keep_prob -- probability that a node will be kept
    training -- boolean to determine if the model is training

    Returns:
    output -- the output of the new layer
    """
    # Create a layer with the given parameters
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer='he_normal')
    # Apply dropout to the layer
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    output = dropout(layer(prev), training=training)
    return output
