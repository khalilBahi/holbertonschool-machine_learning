#!/usr/bin/env python3
"""Task 4. LeNet-5 (Tensorflow)"""
import tensorflow.compat.v1 as tf  # type: ignore


def lenet5(x, y):
    """
    Builds a modified LeNet-5 model using TensorFlow.

    Args:
        x (tf.placeholder): tf.placeholder of shape (m, 28, 28, 1)
        containing the input images for the network.
        y (tf.placeholder): tf.placeholder of shape (m, 10)
        containing the one-hot labels for the network.

    Returns:
        tuple: A tuple containing:
            - y_pred (tf.Tensor): A tensor containing the softmax predictions.
            - train (tf.Operation): An operation that
            trains the network using Adam optimizer.
            - loss (tf.Tensor): A tensor containing the loss of the network.
            - accuracy (tf.Tensor): A tensor containing
            the accuracy of the network.
    """
    # initialize global parameters
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # First convolutional layer
    conv_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=init,
    )(x)

    # First pooling layer
    pool_1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv_1)

    # Second convolutional layer
    conv_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="valid",
        activation="relu",
        kernel_initializer=init,
    )(pool_1)

    # Second pooling layer
    pool_2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv_2)

    # Flatten the output of the second pooling layer
    flat = tf.layers.Flatten()(pool_2)

    # First fully connected layer
    layer_1 = tf.layers.Dense(
        units=120, activation="relu", name="layer", kernel_initializer=init
    )(flat)

    # Second fully connected layer
    layer_2 = tf.layers.Dense(
        units=84, activation="relu", name="layer", kernel_initializer=init
    )(layer_1)

    # Output layer
    output = tf.layers.Dense(
        units=10, activation=None, name="layer", kernel_initializer=init
    )(layer_2)

    # Loss function
    losses = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)

    # Calculate accuracy
    comparation = tf.math.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(comparation, tf.float32))

    # Optimizer
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(losses)

    # Softmax output
    out = tf.nn.softmax(output)

    return out, train, losses, accuracy
