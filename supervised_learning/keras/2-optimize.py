#!/usr/bin/env python3
""" Task 2: 2. Optimize """

import tensorflow as tf
K = tf.keras


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with categorical
    crossentropy loss and accuracy metrics.

    Parameters:
    network (K.Model): The model to optimize.
    alpha (float): The learning rate for the Adam optimizer.
    beta1 (float): The first Adam optimization parameter
    (exponential decay rate for the first moment estimates).
    beta2 (float): The second Adam optimization parameter
    (exponential decay rate for the second moment estimates).

    Returns:
    None
    """
    # Define the Adam optimizer with the specified parameters
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2)

    # Add the 'lr' attribute for backward compatibility
    optimizer.lr = optimizer.learning_rate

    # Compile the model with categorical crossentropy loss and accuracy metrics
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
