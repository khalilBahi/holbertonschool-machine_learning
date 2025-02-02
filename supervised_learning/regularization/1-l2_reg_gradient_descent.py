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
    for i in range(L, 0, -1):
        # Retrieve the weights and biases of the current layer
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        # Calculate the derivative of the activation function
        if i == L:
            dZ = A - Y
        else:
            dZ = np.matmul(weights_copy['W' + str(i + 1)].T, dZ) * (1 - A ** 2)
        # Calculate the gradient of the weights
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        # Calculate the gradient of the biases
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        # Update the weights and biases
        weights_copy['W' + str(i)] = W - alpha * dW
        weights_copy['b' + str(i)] = b - alpha * db
    return weights_copy
