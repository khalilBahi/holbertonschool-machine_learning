#!/usr/bin/env python3
""" Task 1: 1. Inputl """
import tensorflow.keras as K  # type: ignore


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using the Functional API.

    Parameters:
    nx (int): Number of input features to the network.
    layers (list): List containing the number of nodes
    in each layer of the network.
    activations (list): List containing the activation
    functions used for each layer of the network.
    lambtha (float): L2 regularization parameter.
    keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
    model (K.Model): The Keras model.
    """
    # Define the input layer
    inputs = K.Input(shape=(nx,))

    # Initialize the input tensor for the first layer
    x = inputs

    # Add layers to the model
    for i in range(len(layers)):
        # Add a Dense layer with L2 regularization
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        # Add Dropout layer (except after the last layer)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # Create the model
    model = K.Model(inputs=inputs, outputs=x)

    return model
