#!/usr/bin/env python3
""" Task 16: 16. DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network for binary classification.

    Attributes:
    - __L (int): Number of layers in the neural network.
    - __cache (dict): Dictionary to store intermediary
    values during forward propagation.
    - __weights (dict): Dictionary to store weights and biases of the network.
    """

    def __init__(self, nx, layers):
        """
        Class constructor for DeepNeuralNetwork.

        Parameters:
        - nx (int): Number of input features.
        - layers (list): List containing the number of
        nodes in each layer of the network.

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
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.__L + 1):
            layer_size = layers[i - 1]
            prev_layer_size = nx if i == 1 else layers[i - 2]

            # He et al. initialization for weights
            self.__weights[f"W{i}"] = np.random.randn(
                layer_size, prev_layer_size
            ) * np.sqrt(2 / prev_layer_size)
            # Bias initialization to 0
            self.__weights[f"b{i}"] = np.zeros((layer_size, 1))

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

    def forward_prop(self, X):
        """
        Performs forward propagation for the neural network.

        Parameters:
        - X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
        - A (numpy.ndarray): The output of the final
        layer after forward propagation.
        - cache (dict): Updated cache with the activations.
        """
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]
            A_prev = self.__cache[f"A{i-1}"]

            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function

            self.__cache[f"A{i}"] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression.

        Parameters:
        - Y (numpy.ndarray): Correct labels of shape (1, m).
        - A (numpy.ndarray): Activated output of shape (1, m).

        Returns:
        - cost (float): The cost of the model.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Parameters:
        - X (numpy.ndarray): Input data of shape (nx, m).
        - Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
        - prediction (numpy.ndarray): Predicted
        labels for each example (1 or 0).
        - cost (float): Cost of the network.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
