#!/usr/bin/env python3
"""Task 5: 5. Dropout Gradient Descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout
    regularization using gradient descent.

    Parameters:
    Y -- one-hot numpy.ndarray of shape (classes, m)
    containing the correct labels for the data
    weights -- dictionary of the weights and biases of the neural network
    cache -- dictionary of the outputs and dropout
    masks of each layer of the neural network
    alpha -- learning rate
    keep_prob -- probability that a node will be kept
    L -- number of layers of the network
    """
    m = Y.shape[1]
    weights_copy = weights.copy()

    for i in reversed(range(L)):
        A = cache["A" + str(i + 1)]
        if i == L - 1:
            dZ = A - Y
        else:
            dW2 = np.matmul(weights_copy["W" + str(i + 2)].T, dZ)
            dtanh = 1 - (A * A)
            dZ = dW2 * dtanh
            dZ = dZ * cache["D" + str(i + 1)]
            dZ = dZ / keep_prob

        dW = np.matmul(dZ, cache["A" + str(i)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W" + str(i + 1)] = (
            weights_copy["W" + str(i + 1)] - (alpha * dW))
        weights["b" + str(i + 1)] = (
            weights_copy["b" + str(i + 1)] - (alpha * db))
