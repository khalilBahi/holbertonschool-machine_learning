#!/usr/bin/env python3
"""Task 7: DenseNet-121"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely
    Connected Convolutional Networks.
    Arguments:
    growth_rate -- growth rate for the dense block
    compression -- compression factor for the transition layer
    Returns:
    model -- the keras model
    """
    init = K.initializers.he_normal(seed=0)
    input_layer = K.Input(shape=(224, 224, 3))
    nb_filters = 2 * growth_rate
    norm1 = K.layers.BatchNormalization()(input_layer)
    act1 = K.layers.Activation("relu")(norm1)

    conv1 = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=init,
    )(act1)

    max_pool = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(conv1)

    dense1, nb_filters = dense_block(max_pool, nb_filters, growth_rate, 6)
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)

    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)

    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)

    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(
        pool_size=7,
        strides=None,
        padding="same",
    )(dense4)

    output = K.layers.Dense(
        units=1000,
        activation="softmax",
        kernel_initializer=init,
    )(avg_pool)

    model = K.models.Model(inputs=input_layer, outputs=output)

    return model
