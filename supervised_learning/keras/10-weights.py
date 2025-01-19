#!/usr/bin/env python3
""" Task 10: 10. Save and Load Weights """

import tensorflow.keras as K  # type: ignore


def save_weights(network, filename):
    """
    Saves a model's weights to a file.

    Args:
        network (keras.Model): The model whose weights should be saved.
        filename (str): The path of the file where the weights should be saved.

    Returns:
        None
    """
    network.save_weights(filename)
    return None


def load_weights(network, filename):
    """
    Loads a model's weights from a file.

    Args:
        network (keras.Model): The model to which the weights should be loaded.
        filename (str): The path of the file from which the
        weights should be loaded.

    Returns:
        None
    """
    network.load_weights(filename)
    return None
