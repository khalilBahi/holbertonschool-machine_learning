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

        # back private attributes (properties are defined below)
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Combine into a single loop
        prev_nodes = nx
        for i, nodes in enumerate(layers):
            # validate each layer inside the single allowed loop
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            # He et al. initialization: split expression to fit style limits
            self.__weights[f"W{i + 1}"] = (
                np.random.randn(nodes, prev_nodes)
                * np.sqrt(2 / prev_nodes)
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

    @L.setter
    def L(self, value):
        """Allow setting L (used by some test scripts)."""
        self.__L = value

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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network.

        Parameters:
        - Y (numpy.ndarray): Correct labels of shape (1, m).
        - cache (dict): Dictionary of cached intermediary values.
        - alpha (float): Learning rate.
        """
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()

        dZ = cache[f"A{L}"] - Y

        for i in range(L, 0, -1):
            A_prev = cache[f"A{i-1}"]
            W = weights[f"W{i}"]

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                dZ = np.dot(W.T, dZ) * A_prev * (1 - A_prev)

            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Parameters:
        - X (numpy.ndarray): Input data of shape (nx, m).
        - Y (numpy.ndarray): Correct labels of shape (1, m).
        - iterations (int): Number of iterations to train over.
        - alpha (float): Learning rate.

        Returns:
        - prediction (numpy.ndarray): The evaluation
        of the training data after iterations.
        - cost (float): Final cost after training.

        Raises:
        - TypeError: If iterations is not an integer or alpha is not a float.
        - ValueError: If iterations is not positive or alpha is not positive.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be an float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if verbose and (i % step == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)
                steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            try:
                import matplotlib.pyplot as plt

                plt.plot(steps, costs, 'b')
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.title('Training Cost')
                plt.show()
            except Exception:
                pass

        return self.evaluate(X, Y)
