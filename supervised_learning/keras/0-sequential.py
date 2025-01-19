#!/usr/bin/env python3
""" Task 0: 0. Sequential """
import tensorflow.keras as K  # type: ignore


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Parameters:
    nx (int): Number of input features to the network.
    layers (list): List containing the number of
    nodes in each layer of the network.
    activations (list): List containing the activation
    functions used for each layer of the network.
    lambtha (float): L2 regularization parameter.
    keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
    model (K.Model): The Keras model.
    """
    model = K.Sequential()

    # Add the first layer with input shape
    model.add(K.layers.Dense(
        layers[0],
        input_shape=(nx,),
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    ))
    model.add(K.layers.Dropout(1 - keep_prob))

    # Add the remaining layers
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))
        if i < len(layers) - 1:  # No dropout after the last layer
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
