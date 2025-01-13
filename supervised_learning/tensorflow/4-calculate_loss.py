#!/usr/bin/env python3
""" Task 4: 4. Loss """
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y: tf.placeholder, labels of the input data.
        y_pred: tf.Tensor, the network's predictions.

    Returns:
        tf.Tensor: A tensor containing the loss of the prediction.
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
