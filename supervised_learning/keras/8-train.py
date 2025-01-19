#!/usr/bin/env python3
""" Task 6: 6. Early Stopping """
import tensorflow.keras as K  # type: ignore


def lr_decay_function(alpha, decay_rate):
    """
    Creates a function to calculate the learning rate using inverse time decay.

    Parameters:
    alpha (float): The initial learning rate.
    decay_rate (float): The decay rate.

    Returns:
    function: A function that takes the epoch
    index and returns the updated learning rate.
    """
    def decay_fn(epoch):
        return alpha / (1 + decay_rate * epoch)
    return decay_fn


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional early
    stopping, learning rate decay, and saving the best model.

    Parameters:
    network (K.Model): The model to train.
    data (numpy.ndarray): Input data of shape (m, nx).
    labels (numpy.ndarray): One-hot labels of shape (m, classes).
    batch_size (int): Size of the batch used for mini-batch gradient descent.
    epochs (int): Number of passes through the data.
    validation_data (tuple, optional): Tuple of
    (X_valid, Y_valid) for validation.
    early_stopping (bool): Whether to use early stopping.
    patience (int): Number of epochs to wait for
    improvement in validation loss before stopping.
    learning_rate_decay (bool): Whether to use learning rate decay.
    alpha (float): The initial learning rate.
    decay_rate (float): The decay rate for learning rate decay.
    save_best (bool): Whether to save the best model based on validation loss.
    filepath (str): The file path where the model should be saved.
    verbose (bool): Whether to print output during training.
    shuffle (bool): Whether to shuffle the batches every epoch.

    Returns:
    K.callbacks.History: The History object generated after training the model.
    """
    callbacks = []

    # Early Stopping Callback
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=patience,   # Number of epochs to wait before stopping
            restore_best_weights=True  # Restore the best model weights
        )
        callbacks.append(early_stopping_callback)

    # Learning Rate Decay Callback
    if learning_rate_decay and validation_data is not None:
        lr_decay_callback = K.callbacks.LearningRateScheduler(
            # Learning rate decay function
            lr_decay_function(alpha, decay_rate),
            verbose=1  # Print a message when the learning rate is updated
        )
        callbacks.append(lr_decay_callback)

    # Save Best Model Callback
    if save_best and validation_data is not None and filepath is not None:
        checkpoint_callback = K.callbacks.ModelCheckpoint(
            filepath=filepath,  # File path to save the model
            monitor='val_loss',  # Monitor validation loss
            save_best_only=True,  # Save only the best model
            mode='min',  # Save the model with the minimum validation loss
            verbose=1  # Print a message when the model is saved
        )
        callbacks.append(checkpoint_callback)

    # Train the model
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
