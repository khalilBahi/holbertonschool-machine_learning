#!/usr/bin/env python3
"""27. Update DeepNeuralNetwork"""

import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing multiclass classification.
    """

    def __init__(self, nx, layers):
        """
        Initialize the DeepNeuralNetwork instance.

        Args:
            nx (int): Number of input features.
            layers (list): List containing the number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is
            not a list of positive integers.
            ValueError: If nx < 1 or layers is empty.
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

        # He initialization
        prev_nodes = nx
        for i, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            self.__weights[f"W{i + 1}"] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights[f"b{i + 1}"] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        """Returns the number of layers."""
        return self.__L

    @L.setter
    def L(self, value):
        """Sets the number of layers (used for updates or loading models)."""
        self.__L = value

    @property
    def cache(self):
        """Returns the cache dictionary that
        stores intermediate activations."""
        return self.__cache

    @property
    def weights(self):
        """Returns the weights and biases dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """
        Performs forward propagation through the network.

        Args:
            X (ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (A_last, cache)
                - A_last: Final layer activation (softmax output probabilities)
                - cache: Dictionary of all activations per layer
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            A_prev = self.__cache[f'A{i-1}']
            Z = np.dot(W, A_prev) + b

            # Hidden layers: sigmoid
            if i != self.__L:
                A = 1 / (1 + np.exp(-Z))
            else:
                # Output layer: softmax
                Z_shift = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_shift)
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            self.__cache[f'A{i}'] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        Computes the categorical cross-entropy cost.

        Args:
            Y (ndarray): True labels (one-hot encoded).
            A (ndarray): Output probabilities from softmax.

        Returns:
            float: The cross-entropy loss.
        """
        m = Y.shape[1]
        A_clipped = np.clip(A, 1e-8, 1 - 1e-8)
        cost = -np.sum(Y * np.log(A_clipped)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the model’s predictions and computes the cost.

        Args:
            X (ndarray): Input data.
            Y (ndarray): True labels (one-hot encoded).

        Returns:
            tuple: (predictions, cost)
                - predictions: One-hot encoded predictions.
                - cost: Computed cost value.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        preds = np.argmax(A, axis=0)
        one_hot = np.zeros_like(A)
        one_hot[preds, np.arange(A.shape[1])] = 1

        return one_hot, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent to update weights and biases.

        Args:
            Y (ndarray): True labels (one-hot encoded).
            cache (dict): Cached activations from forward propagation.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()

        dZ = cache[f'A{L}'] - Y  # Output layer derivative
        for i in range(L, 0, -1):
            A_prev = cache[f'A{i-1}']
            W = weights[f'W{i}']

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                dZ = np.dot(W.T, dZ) * A_prev * (1 - A_prev)

            # Parameter update
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Args:
            X (ndarray): Input data.
            Y (ndarray): One-hot encoded true labels.
            iterations (int): Number of iterations for training.
            alpha (float): Learning rate.
            verbose (bool): If True, prints cost every 'step' iterations.
            graph (bool): If True, plots cost vs. iterations graph.
            step (int): Frequency of printing cost during training.

        Returns:
            tuple: (predictions, cost)
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

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.

        Args:
            filename (str): Path to the file to save the model.
        """
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except Exception:
            return False

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.

        Args:
            filename (str): Path to the .pkl file.

        Returns:
            DeepNeuralNetwork: Loaded model instance,
            or None if file not found.
        """
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except Exception:
            return None
