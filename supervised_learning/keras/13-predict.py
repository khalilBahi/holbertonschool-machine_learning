#!/usr/bin/env python3
""" Task 13: 13. Predict """
import tensorflow.keras as K  # type: ignore


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network (keras.Model): The network model to make the prediction with.
        data (numpy.ndarray): The input data to make the prediction with.
        verbose (bool): Whether to print output during the prediction process.

    Returns:
        numpy.ndarray: The prediction for the data.
    """
    # Use the network's predict method to generate predictions
    predictions = network.predict(data, verbose=verbose)

    return predictions
