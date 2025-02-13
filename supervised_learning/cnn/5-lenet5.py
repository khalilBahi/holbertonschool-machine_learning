#!/usr/bin/env python3
"""Task 5. LeNet-5 (Keras)"""
import tensorflow.keras as K  # type: ignore


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
    # initialize global parameters
    init = K.initializers.he_normal()

    # First CONVNET
    conv1 = K.layers.Conv2D(filters=6, kernel_size=5,
                            padding='same', activation='relu',
                            kernel_initializer=init)(X)
    # Pool net of CONV1
    pool1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)

    # Second CONVNET
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5,
                            padding='valid', activation='relu',
                            kernel_initializer=init)(pool1)
    # Pool net of CONV2
    pool2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # Flatten the convolutional layers
    flatten = K.layers.Flatten()(pool2)

    # Fully connected layer 1
    FC1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=init)(flatten)
    # Fully connected layer 2
    FC2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=init)(FC1)
    # Fully connected layer 3
    FC3 = K.layers.Dense(units=10, kernel_initializer=init,
                         activation='softmax')(FC2)

    # Create Model
    model = K.models.Model(X, FC3)

    # Set Adam optimizer
    adam = K.optimizers.Adam()

    # Compile model
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
