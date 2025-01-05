#!/usr/bin/env python3
""" Task 8: 8. NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """
    Class representing a neural network with one
    hidden layer for binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initialize a NeuralNetwork instance.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for weights vector of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Getter for bias of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Getter for activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Getter for weights vector of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Getter for bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Getter for activated output of the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: Activated outputs A1 and A2.
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels for the input data (1, m).
            A (numpy.ndarray): Activated output of the neuron for each
            example (1, m).

        Returns:
            float: The cost.
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels for the input data (1, m).

        Returns:
            tuple: Prediction and cost of the network.
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost
