#!/usr/bin/env python3
""" Task 16: 16. DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork.

        Parameters:
        - nx: Number of input features.
        - layers: List representing the number of nodes in each layer.

        Raises:
        - TypeError: If nx is not an integer or
          layers is not a list of positive integers.
        - ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # initialize private attributes to back the properties
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Combine into a single loop
        prev_nodes = nx
        for i, nodes in enumerate(layers):
            # validate each layer inside the single allowed loop
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            self.__weights[f"W{i + 1}"] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights[f"b{i + 1}"] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        """
        Getter for the number of layers.

        Returns:
        int: Number of layers in the network.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter for the cache dictionary.

        Returns:
        dict: Cache dictionary containing intermediary values of the network.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter for the weights dictionary.

        Returns:
        dict: Weights dictionary containing weights and biases of the network.
        """
        return self.__weights
