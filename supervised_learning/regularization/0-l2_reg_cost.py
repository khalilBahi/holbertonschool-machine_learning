#!/usr/bin/env python3
"""Task 0: 0. L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost -- cost of the network without L2 regularization
    lambtha -- regularization parameter
    weights -- dictionary of the weights and biases
    (numpy.ndarrays) of the neural network
    L -- number of layers in the neural network
    m -- number of data points used

    Returns:
    cost -- the cost of the network accounting for L2 regularization
    """
    norm = 0

    # Loop through all layers and sum the squares of the weight
    for i in range(1, L + 1):
        norm += np.linalg.norm(weights['W' + str(i)])

    # Add the L2 regularization term to the original cost
    l2_cost = cost + (lambtha / (2 * m)) * norm

    return l2_cost
