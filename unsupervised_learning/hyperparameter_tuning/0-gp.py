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
