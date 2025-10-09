#!/usr/bin/env python3
"""26. Persistence is Key"""

import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.

    Attributes
    ----------
    __L : int
        The number of layers in the neural network.
    __cache : dict
        Dictionary holding all intermediary values of the network during
        forward propagation.
    __weights : dict
        Dictionary containing all weights and bias matrices for each layer.

    Methods
    -------
    forward_prop(X):
        Performs forward propagation through the network.

    cost(Y, A):
        Computes the logistic regression cost function.

    evaluate(X, Y):
        Evaluates the network’s predictions and cost.

    gradient_descent(Y, cache, alpha=0.05):
        Performs one pass of backpropagation and updates the weights and biases.

    train(X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        Trains the neural network with gradient descent and optionally displays cost progression.

    save(filename):
        Saves the trained model to disk in `.pkl` format.

    load(filename):
        Static method to load a previously saved model.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Parameters
        ----------
        nx : int
            Number of input features.
        layers : list
            List representing the number of nodes in each layer of the
            network.

        Raises
        ------
        TypeError
            If `nx` is not an integer or `layers` is not a list of positive
            integers.
        ValueError
            If `nx` is less than 1 or any layer size is less than 1.

        Notes
        -----
        - Weights are initialized using He initialization:
          Wᵢ ~ N(0, sqrt(2 / prev_layer_nodes))
        - Biases are initialized as zeros.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_nodes = nx
        for i, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            self.__weights[f"W{i + 1}"] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights[f"b{i + 1}"] = np.zeros((nodes, 1))
            prev_nodes = nodes

    # -------------------- Properties --------------------

    @property
    def L(self):
        """int: Number of layers in the network."""
        return self.__L

    @property
    def cache(self):
        """dict: Stores intermediary values (activations, inputs) during
        forward propagation.
        """
        return self.__cache

    @property
    def weights(self):
        """dict: Contains all the network’s weights and biases."""
        return self.__weights

    # -------------------- Core Computation Methods --------------------

    def forward_prop(self, X):
        """
        Performs forward propagation through the neural network.

        Parameters
        ----------
        X : ndarray
            Input data of shape (nx, m), where:
            - nx = number of features
            - m = number of examples

        Returns
        -------
        A : ndarray
            Final activation output (predicted probabilities).
        cache : dict
            Dictionary storing all intermediate activations and linear
            combinations (Z).
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            A_prev = self.__cache[f'A{i-1}']
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f'A{i}'] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        Computes the logistic regression cost.

        Parameters
        ----------
        Y : ndarray
            True labels of shape (1, m).
        A : ndarray
            Predicted probabilities of shape (1, m).

        Returns
        -------
        float
            The logistic regression cost.
        """
        m = Y.shape[1]
        cost = -np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        ) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Parameters
        ----------
        X : ndarray
            Input data.
        Y : ndarray
            True labels.

        Returns
        -------
        prediction : ndarray
            Binary predictions (0 or 1).
        cost : float
            Computed cost of the evaluation.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the network.

        Parameters
        ----------
        Y : ndarray
            True labels.
        cache : dict
            Cached forward propagation values.
        alpha : float, optional
            Learning rate (default = 0.05).

        Notes
        -----
        - Uses backpropagation to compute gradients.
        - Updates weights and biases in place.
        """
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()

        dZ = cache[f'A{L}'] - Y
        for i in range(L, 0, -1):
            A_prev = cache[f'A{i-1}']
            W = weights[f'W{i}']
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 1:
                dZ = (
                    np.dot(W.T, dZ) * A_prev * (1 - A_prev)
                )
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db

    # -------------------- Training Method --------------------

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neural network using gradient descent.

        Parameters
        ----------
        X : ndarray
            Input data.
        Y : ndarray
            True labels.
        iterations : int, optional
            Number of training iterations (default = 5000).
        alpha : float, optional
            Learning rate (default = 0.05).
        verbose : bool, optional
            If True, prints cost every `step` iterations (default = True).
        graph : bool, optional
            If True, plots the cost curve (default = True).
        step : int, optional
            Step interval for printing/logging cost (default = 100).

        Returns
        -------
        tuple
            (prediction, cost) after the final iteration.

        Raises
        ------
        TypeError
            If input arguments are of invalid types.
        ValueError
            If numerical parameters are invalid.
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
                raise ValueError(
                    "step must be positive and <= iterations"
                )

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

    # -------------------- Persistence Methods --------------------

    def save(self, filename):
        """
        Saves the current network instance to a pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to save (will add `.pkl` if missing).

        Returns
        -------
        bool
            True if saved successfully, False otherwise.
        """
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            return True
        except Exception:
            return False

    @staticmethod
    def load(filename):
        """
        Loads a previously saved DeepNeuralNetwork object.

        Parameters
        ----------
        filename : str
            Path to the pickle file.

        Returns
        -------
        DeepNeuralNetwork or None
            Loaded object if successful, None otherwise.
        """
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except Exception:
            return None
