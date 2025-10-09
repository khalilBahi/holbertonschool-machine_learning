#!/usr/bin/env python3
"""Task 16: DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork.

        Parameters:
        - nx: number of input features
        - layers: list representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_nodes = nx
        for i, nodes in enumerate(layers):
            self.__weights[f"W{i + 1}"] = np.random.randn
            (nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.__weights[f"b{i + 1}"] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Dictionary storing intermediate values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Dictionary storing weights and biases of the network"""
        return self.__weights
