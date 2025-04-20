#!/usr/bin/env python3
"""Task 0. Initialize Gaussian Processs"""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian Process.

        Args:
            X_init (numpy.ndarray): Shape (t, 1),
            inputs sampled with black-box function.
            Y_init (numpy.ndarray): Shape (t, 1),
            outputs of black-box function for X_init.
            l (float): Length parameter for the kernel. Default is 1.
            sigma_f (float): Standard deviation of
            the black-box function output. Default is 1.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculate the covariance kernel matrix using RBF kernel.

        Args:
            X1 (numpy.ndarray): Shape (m, 1), first set of inputs.
            X2 (numpy.ndarray): Shape (n, 1), second set of inputs.

        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n).
        """
        # Compute squared Euclidean distance: ||x1 - x2||^2
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
            np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        # RBF kernel: sigma_f^2 * exp(-0.5 * ||x1 - x2||^2 / l^2)
        return self.sigma_f**2 * np.exp(-0.5 * sqdist / self.l**2)

    def predict(self, X_s):
        """Predict the mean and variance of points in a Gaussian process.

        Args:
            X_s (numpy.ndarray): Shape (s, 1),
            points to predict mean and variance for.

        Returns:
            tuple: (mu, sigma)
                - mu (numpy.ndarray): Shape (s,),
                predicted mean for each point in X_s.
                - sigma (numpy.ndarray): Shape (s,),
                predicted variance for each point in X_s.
        """
        # Compute covariance matrices
        K_s = self.kernel(X_s, self.X)  # Shape (s, t)
        K_ss = self.kernel(X_s, X_s)    # Shape (s, s)

        # Solve for K(X, X)^{-1} Y and K(X, X)^{-1} K(X, X_s)
        K_inv = np.linalg.inv(self.K)   # Shape (t, t)
        mu = K_s @ K_inv @ self.Y       # Shape (s, 1)
        cov = K_ss - K_s @ K_inv @ K_s.T  # Shape (s, s)

        # Extract mean and variance
        mu = mu.flatten()               # Shape (s,)
        sigma = np.diag(cov)            # Shape (s,)

        return mu, sigma

    def update(self, X_new, Y_new):
        """Update the Gaussian Process with a new sample point.

        Args:
            X_new (numpy.ndarray): Shape (1,), new sample point.
            Y_new (numpy.ndarray): Shape (1,), new sample function value.
        """
        # Reshape X_new and Y_new to (1, 1) for consistency
        X_new = X_new.reshape(1, 1)
        Y_new = Y_new.reshape(1, 1)

        # Append new point to X and Y
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        # Recompute the covariance matrix K
        self.K = self.kernel(self.X, self.X)
