#!/usr/bin/env python3
""" Task 6: 6. Early Stopping """
import tensorflow.keras as K  # type: ignore


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    with optional early stopping.

    Parameters:
    network (K.Model): The model to train.
    data (numpy.ndarray): Input data of shape (m, nx).
    labels (numpy.ndarray): One-hot labels of shape (m, classes).
    batch_size (int): Size of the batch used for mini-batch gradient descent.
    epochs (int): Number of passes through the data.
    validation_data (tuple, optional):
    Tuple of (X_valid, Y_valid) for validation.
    early_stopping (bool): Whether to use early stopping.
    patience (int): Number of epochs to wait
    for improvement in validation loss before stopping.
    verbose (bool): Whether to print output during training.
    shuffle (bool): Whether to shuffle the batches every epoch.

    Returns:
    K.callbacks.History: The History object generated after training the model.
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=patience,   # Number of epochs to wait before stopping
            restore_best_weights=True  # Restore the best model weights
        )
        callbacks.append(early_stopping_callback)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
    return history
