#!/usr/bin/env python3
""" Task 0: 0. Neuron"""
import numpy as np


class Neuron:
    """
    Class representing a single neuron for binary classification.
    """

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for bias."""
        return self.__b

    @property
    def A(self):
        """Getter for activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            numpy.ndarray: Activated output (A) of the neuron.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
            float: Cost of the model.
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron’s predictions.

        Args:
            X (numpy.ndarray): Input data (nx, m)
            Y (numpy.ndarray): Correct labels (1, m)

        Returns:
            tuple: Prediction (1, m) and cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
