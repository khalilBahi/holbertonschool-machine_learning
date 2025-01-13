#!/usr/bin/env python3
""" Task 5: 5. Train_Op """
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Args:
        loss: tf.Tensor, the loss of the network's prediction.
        alpha: float, the learning rate.

    Returns:
        tf.Operation: An operation that trains the network.
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
