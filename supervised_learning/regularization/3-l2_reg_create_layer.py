#!/usr/bin/env python3
"""Task 3: 3. Create a Layer with L2 Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow
    that includes L2 regularization.

    Parameters:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function that should be used on the layer
    lambtha -- L2 regularization parameter

    Returns:
    output -- the output of the new layer
    """
    # Create a kernel initializer with L2 regularization
    regularizer = tf.keras.regularizers.l2(lambtha)
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    # Create a layer with the given parameters
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)
    return layer(prev)
