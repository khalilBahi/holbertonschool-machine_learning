#!/usr/bin/env python3
"""Task 5. LeNet-5 (Keras)"""
from tensorflow import keras as K  # type: ignore


def lenet5(X):
    """
    Builds a modified LeNet-5 model using Keras.

    Args:
        X (K.Input): Keras Input layer of shape (m, 28, 28, 1)
        containing the input images for the network.

    Returns:
        K.Model: A Keras Model compiled to use Adam
        optimizer and categorical crossentropy loss.
    """
    # Initialize the HeNormal initializer
    init = K.initializers.HeNormal(seed=0)

    # Define the model architecture
    model = K.Sequential(
        [
            X,
            # First convolutional layer
            K.layers.Conv2D(
                filters=6,
                kernel_size=5,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # First pooling layer
            K.layers.MaxPool2D(pool_size=2, strides=2),
            # Second convolutional layer
            K.layers.Conv2D(
                filters=16, kernel_size=5, activation="relu",
                kernel_initializer=init
            ),
            # Second pooling layer
            K.layers.MaxPool2D(pool_size=2, strides=2),
            # Flatten the output
            K.layers.Flatten(),
            # First fully connected layer
            K.layers.Dense(
                units=120,
                activation="relu",
                kernel_initializer=init),
            # Second fully connected layer
            K.layers.Dense(
                units=84,
                activation="relu",
                kernel_initializer=init),
            # Output layer
            K.layers.Dense(
                units=10,
                activation="softmax",
                kernel_initializer=init),
        ]
    )

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=K.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model
