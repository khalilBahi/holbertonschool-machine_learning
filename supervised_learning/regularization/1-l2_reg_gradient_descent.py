#!/usr/bin/env python3
"""Task 1: 1. Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network
    using gradient descent with L2 regularization.

    Parameters:
    Y -- one-hot numpy.ndarray of shape (classes, m)
    containing the correct labels for the data
    weights -- dictionary of the weights and biases of the neural network
    cache -- dictionary of the outputs of each layer of the neural network
    alpha -- learning rate
    lambtha -- L2 regularization parameter
    L -- number of layers of the network

    Returns:
    None (weights and biases are updated in place)
    """
    # Calculate the number of data points
    m = Y.shape[1]
    # Copy the weights dictionary
    weights_copy = weights.copy()
    # Loop through all layers in reverse order
    for i in reversed(range(L)):
        A = cache["A" + str(i + 1)]
        if i == L - 1:
            dZ = cache["A" + str(i + 1)] - Y
        else:
            dW2 = np.matmul(weights_copy["W" + str(i + 2)].T, dZ)
            tanh = 1 - (A * A)
            dZ = dW2 * tanh

        dW = np.matmul(dZ, cache["A" + str(i)].T) / m
        dW_L2 = dW + (lambtha / m) * weights_copy["W" + str(i + 1)]
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W" + str(i + 1)] = (
            weights_copy["W" + str(i + 1)] - (alpha * dW_L2))
        weights["b" + str(i + 1)] = (
            weights_copy["b" + str(i + 1)] - (alpha * db))
