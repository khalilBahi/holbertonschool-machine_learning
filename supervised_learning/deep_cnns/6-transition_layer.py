#!/usr/bin/env python3
"""Task 6: Transition Layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely
    Connected Convolutional Networks.

    Arguments:
    X -- output from the previous layer
    nb_filters -- integer representing the number of filters in X
    compression -- compression factor for the transition layer

    Returns:
    Y -- output from the transition layer
    nb_filters -- number of filters within the transition layer
    """
    init = K.initializers.he_normal(seed=0)
    filters = int(nb_filters * compression)

    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation("relu")(norm1)

    conv1 = K.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding="same",
        kernel_initializer=init,
    )(act1)

    avg_pool = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        padding="same",
    )(conv1)

    return avg_pool, filters
