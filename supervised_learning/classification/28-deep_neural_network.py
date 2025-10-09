#!/usr/bin/env python3
"""28. All the Activations"""

import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """
    Defines a deep neural network for multiclass classification.

    Attributes:
        __L (int): Number of layers in the network.
        __cache (dict): Stores intermediary values
        (A and Z) during forward propagation.
        __weights (dict): Stores weights and biases for each layer.
        __activation (str): Activation function used
        in hidden layers ('sig' or 'tanh').
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List containing the number of nodes in each layer.
            activation (str): Activation for hidden layers. Either:
                - 'sig' for sigmoid
                - 'tanh' for hyperbolic tangent

        Raises:
            TypeError: If nx is not an integer or layers
            is not a list of positive integers.
            ValueError: If nx < 1 or activation not in {'sig', 'tanh'}.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights using He initialization
        prev_nodes = nx
        for i, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            # He initialization ensures better convergence
            # for ReLU-like activations,
            # but still performs well with sigmoid/tanh
            self.__weights[f"W{i + 1}"] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights[f"b{i + 1}"] = np.zeros((nodes, 1))
            prev_nodes = nodes

    # -------------------- Properties -------------------- #

    @property
    def L(self):
        """Number of layers in the network."""
        return self.__L

    @property
    def cache(self):
        """Dictionary containing cached activations and linear transforms."""
        return self.__cache

    @property
    def weights(self):
        """Dictionary containing weights and biases of each layer."""
        return self.__weights

    @property
    def activation(self):
        """Activation function type for hidden layers ('sig' or 'tanh')."""
        return self.__activation

    # -------------------- Activation Helpers -------------------- #

    def __hidden_activate(self, Z):
        """Apply the selected activation function
        to hidden layer outputs (Z)."""
        if self.__activation == 'sig':
            return 1 / (1 + np.exp(-Z))  # Sigmoid
        return np.tanh(Z)  # Tanh

    def __hidden_derivative(self, A):
        """Compute the derivative of the hidden
        activation for backpropagation."""
        if self.__activation == 'sig':
            return A * (1 - A)
        return 1 - (A ** 2)

    # -------------------- Forward Propagation -------------------- #

    def forward_prop(self, X):
        """
        Performs forward propagation through the network.

        Hidden layers: use chosen activation (sigmoid or tanh)
        Output layer: uses softmax activation

        Args:
            X (numpy.ndarray): Input data of shape (nx, m)

        Returns:
            tuple: (A_last, cache)
                - A_last: output probabilities
                after softmax (shape: classes × samples)
                - cache: dictionary of intermediate activations
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]
            A_prev = self.__cache[f"A{i-1}"]

            Z = np.dot(W, A_prev) + b

            # Hidden layers
            if i != self.__L:
                A = self.__hidden_activate(Z)
            else:
                # Output layer uses softmax (with stability)
                Z_shift = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_shift)
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            self.__cache[f"A{i}"] = A

        return A, self.__cache

    # -------------------- Cost Function -------------------- #

    def cost(self, Y, A):
        """
        Compute the categorical cross-entropy cost.

        Args:
            Y (numpy.ndarray): True one-hot labels (shape: classes × samples)
            A (numpy.ndarray): Predicted probabilities
            (shape: classes × samples)

        Returns:
            float: cost value
        """
        m = Y.shape[1]
        A_clipped = np.clip(A, 1e-8, 1 - 1e-8)
        return -np.sum(Y * np.log(A_clipped)) / m

    # -------------------- Evaluation -------------------- #

    def evaluate(self, X, Y):
        """
        Evaluate network performance.

        Args:
            X (numpy.ndarray): Input data
            Y (numpy.ndarray): True one-hot labels

        Returns:
            tuple: (predictions, cost)
                - predictions: one-hot encoded predictions
                - cost: final cost value
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        preds = np.argmax(A, axis=0)
        one_hot = np.zeros_like(A)
        one_hot[preds, np.arange(A.shape[1])] = 1

        return one_hot, cost

    # -------------------- Backpropagation -------------------- #

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent to update weights.

        Args:
            Y (numpy.ndarray): True labels (one-hot)
            cache (dict): Cached activations from forward propagation
            alpha (float) Learning rate
        """
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()

        # Derivative for softmax + cross-entropy
        dZ = cache[f"A{L}"] - Y

        for i in range(L, 0, -1):
            A_prev = cache[f"A{i-1}"]
            W = weights[f"W{i}"]

            # Compute gradients
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Compute next dZ before updating weights (match task 27 order)
            if i > 1:
                dA_prev = np.dot(W.T, dZ)
                dZ_next = dA_prev * self.__hidden_derivative(A_prev)

            # Update parameters
            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

            # Move to previous layer
            if i > 1:
                dZ = dZ_next

    # -------------------- Training -------------------- #

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the neural network.

        Args:
            X (numpy.ndarray): Input data
            Y (numpy.ndarray): True labels (one-hot)
            iterations (int): Number of training iterations
            alpha (float): Learning rate
            verbose (bool): Whether to print cost updates
            graph (bool): Whether to plot cost curve
            step (int): Interval for logging cost

        Returns:
            tuple: (predictions, final_cost)
        """
        # Match task 27 validation semantics exactly
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

        costs, steps = [], []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            # Logging
            if verbose and (i % step == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)
                steps.append(i)

            # Update weights (skip last iteration)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        # Optional graph of cost over time
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

    # -------------------- Model Persistence -------------------- #

    def save(self, filename):
        """
        Save the model instance to a pickle file.

        Args:
            filename (str): File path to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            return True
        except Exception:
            return False

    @staticmethod
    def load(filename):
        """
        Load a saved DeepNeuralNetwork instance from file.

        Args:
            filename (str): Path to the pickle file

        Returns:
            DeepNeuralNetwork or None: The loaded model,
            or None if loading fails
        """
        if not os.path.exists(filename):
            return None

        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
