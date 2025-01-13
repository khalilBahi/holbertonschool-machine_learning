#!/usr/bin/env python3
""" Task 3: 3. Accuracy """
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y: tf.placeholder, labels of the input data.
        y_pred: tf.Tensor, the network’s predictions.

    Returns:
        tf.Tensor: A tensor containing the decimal accuracy of the prediction.
    """
    correct_predictions = tf.equal(
        tf.argmax(
            y, axis=1), tf.argmax(
            y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
