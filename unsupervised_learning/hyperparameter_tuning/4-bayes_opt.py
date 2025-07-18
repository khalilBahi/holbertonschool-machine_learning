#!/usr/bin/env python3
"""task 3. Initialize Bayesian Optimization"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """Calculate the next best sample point using Expected Improvement.

        Uses the Expected Improvement (EI)
        acquisition function to balance exploration
        and exploitation, determining the next point to sample.

        Returns:
            tuple: (X_next, EI)
                - X_next (numpy.ndarray): Shape (1,), next best sample point.
                - EI (numpy.ndarray): Shape (ac_samples,),
                expected improvement for each sample.
        """
        # Predict mean at training points (unused in EI calculation)
        mu_sample, _ = self.gp.predict(self.gp.X)

        # Predict mean and variance for acquisition sample points
        mu, sigma = self.gp.predict(self.X_s)
        # Note: sigma is variance here; EI formula expects standard deviation
        # (np.sqrt(sigma))

        # Handle division by zero warnings during EI calculation
        with np.errstate(divide='warn'):
            if self.minimize is True:
                # Find the best (minimum) observed function value for
                # minimization
                Y_s_opt = np.min(self.gp.Y)
                # Calculate improvement: best value minus predicted mean,
                # adjusted by xsi
                imp = Y_s_opt - mu - self.xsi
            else:
                # Find the best (maximum) observed function value for
                # maximization
                mu_sample_opt = np.max(self.gp.Y)
                # Calculate improvement: predicted mean minus best value,
                # adjusted by xsi
                imp = mu - mu_sample_opt - self.xsi

            # Standardize improvement for normal distribution
            # Warning: This assumes sigma is standard deviation, but it's
            # variance from gp.predict
            Z = imp / sigma

            # Compute EI: (improvement * CDF) + (sigma * PDF)
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Set EI to 0 where sigma is 0 (no uncertainty, e.g., at training
            # points)
            EI[sigma == 0.0] = 0.0

        # Select the acquisition point with maximum EI
        X_next = self.X_s[np.argmax(EI)]

        # Return X_next (shape (1,)) and EI (shape (ac_samples,))
        return X_next, EI.reshape(-1)
