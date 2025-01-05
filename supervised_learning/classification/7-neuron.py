#!/usr/bin/env python3
""" Task 0: 0. Neuron"""
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            A (numpy.ndarray): Activated output.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(
        self, X, Y, iterations=5000, alpha=0.05,
        verbose=True, graph=True, step=100
    ):
        """
        Trains the neuron.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.
            verbose (bool): Print training progress.
            graph (bool): Plot training cost.
            step (int): Step size for printing and plotting.

        Raises:
            TypeError: If iterations is not an integer or alpha is not a float.
            ValueError: If iterations or alpha are not positive.
            TypeError: If step is not an integer.
            ValueError: If step is not positive or greater than iterations.

        Returns:
            tuple: Prediction and cost after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if (verbose or graph) and i % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            plt.plot(range(0, iterations, step), costs, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
