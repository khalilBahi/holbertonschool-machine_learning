#!/usr/bin/env python3
"""Task 2: 2. L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost -- tensor containing the cost of the
    network without L2 regularization.
    model -- tf.keras.Model
    Returns:
    cost -- tensor containing the cost of the
    network accounting for L2 regularization.
    """
    l2_cost = list()
    for layer in model.layers:
        l2_cost.append(tf.reduce_sum(layer.losses) + cost)

    return tf.stack(l2_cost[1:])
