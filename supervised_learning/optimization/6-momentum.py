#!/usr/bin/env python3
"""Task 6: 6. Momentum Upgraded"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum
    optimization algorithm in TensorFlow.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        optimizer: A TensorFlow optimizer
        configured for gradient descent with momentum.
    """
    # Create an SGD optimizer with momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
