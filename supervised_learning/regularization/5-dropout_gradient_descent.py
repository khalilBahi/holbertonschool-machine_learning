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
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        dW = (np.matmul(dz, A.T) / m)
        db = (np.sum(dz, axis=1, keepdims=True) / m)
        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db
        if i > 1:
            dz = np.matmul(W.T, dz) * (A * (1 - A)) * \
                cache["D" + str(i - 1)] / keep_prob
