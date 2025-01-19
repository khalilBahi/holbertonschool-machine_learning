#!/usr/bin/env python3
""" Task 12: 12. Test """
import tensorflow.keras as K  # type: ignore


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Args:
        network (keras.Model): The network model to test.
        data (numpy.ndarray): The input data to test the model with.
        labels (numpy.ndarray): The correct one-hot labels of the data.
        verbose (bool): Whether to print output during the testing process.

    Returns:
        tuple: A tuple containing the loss and accuracy
        of the model with the testing data.
    """
    return network.evaluate(data, labels, verbose=verbose)
