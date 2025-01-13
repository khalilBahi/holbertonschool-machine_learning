#!/usr/bin/env python3
""" Task 0: 0. Placeholders"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Creates placeholders for a neural network.

    Args:
        nx: int, the number of feature columns in our data.
        classes: int, the number of classes in our classifier.

    Returns:
        tuple: placeholders named x and y, respectively.
        x: placeholder for the input data to the neural network.
        y: placeholder for the one-hot labels for the input data.
    """
    x = tf.placeholder(float, shape=[None, nx], name='x')
    y = tf.placeholder(float, shape=[None, classes], name='y')

    return (x, y)
