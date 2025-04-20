#!/usr/bin/env python3
"""task 3. Initialize Bayesian Optimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize the Bayesian Optimization.
a
        Args:
            f (callable): Black-box function to optimize.
            X_init (numpy.ndarray): Shape (t, 1), initial input samples.
            Y_init (numpy.ndarray): Shape (t, 1), outputs of f for X_init.
            bounds (tuple): (min, max), bounds of the search space.
            ac_samples (int): Number of acquisition sample points.
            l (float): Length parameter for the kernel. Default is 1.
            sigma_f (float): Standard deviation of the
            black-box function output. Default is 1.
            xsi (float): Exploration-exploitation factor.
            Default is 0.01.
            minimize (bool): True for minimization, False
            for maximization. Default is True.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
