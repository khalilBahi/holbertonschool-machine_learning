#!/usr/bin/env python3
""" Task 3: 3. One Hot """
import tensorflow.keras as K  # type: ignore


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix using Keras.

    Parameters:
    labels (numpy.ndarray): The label vector to convert.
    classes (int, optional): The number of classes.
    If None, it is inferred from the labels.

    Returns:
    numpy.ndarray: The one-hot matrix.
    """
    # Use Keras's to_categorical function to create the one-hot matrix
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)

    return one_hot_matrix
