#!/usr/bin/env python3
""" Task 2: 2. Forward Propagation """
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x: tf.placeholder, input data.
        layer_sizes: list of int, number of nodes in each layer of the network.
        activations: list of activation functions
        for each layer of the network.

    Returns:
        tf.Tensor: The prediction of the network.
    """
    output = x
    for size, activation in zip(layer_sizes, activations):
        output = create_layer(output, size, activation)  # Removed layer_name
    return output
