#!/usr/bin/env python3
""" Task 4: 4. Train """
import tensorflow.keras as K  # type: ignore


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters:
    network (K.Model): The model to train.
    data (numpy.ndarray): Input data of shape (m, nx).
    labels (numpy.ndarray): One-hot labels of shape (m, classes).
    batch_size (int): Size of the batch used for mini-batch gradient descent.
    epochs (int): Number of passes through the data.
    verbose (bool): Whether to print output during training.
    shuffle (bool): Whether to shuffle the batches every epoch.

    Returns:
    K.callbacks.History: The History object generated after training the model.
    """
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
