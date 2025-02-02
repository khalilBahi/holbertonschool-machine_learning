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
    # Get the weights of the model
    weights = model.trainable_variables
    # Calculate the L2 regularization term
    l2_cost = cost
    for w in weights:
        l2_cost += tf.nn.l2_loss(w)
    return l2_cost
