#!/usr/bin/env python3
"""
Normal distribution
"""

class Normal:
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize a Normal distribution.

        Parameters:
        - data (list, optional): List of data to estimate the distribution.
        - mean (float, optional): Mean of the distribution (default is 0.0).
        - stddev (float, optional): Standard deviation of the distribution (default is 1.0).

        Sets the instance attributes:
        - mean (float): The mean of the Normal distribution.
        - stddev (float): The standard deviation of the Normal distribution.

        Raises:
        - ValueError: If stddev is not positive or if data does not contain multiple values.
        - TypeError: If data is not a list.
        """
        self.e = 2.7182818285
        self.pi = 3.1415926536

        if data is not None:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            new_list = []
            for i in data:
                new_list.append((i-self.mean)**2)
            self.stddev = (sum(new_list) / len(data)) ** 0.5
        else:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = mean
            self.stddev = stddev
    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value of a given z-score.

        Parameters:
        - z (float): The z-score.

        Returns:
        - float: The x-value corresponding to z.
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculate the Probability Density Function (PDF) for a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The PDF value for x (probability density at x).
        """
        # Define constants
        e = 2.7182818285
        pi = 3.1415926536
        
        # Calculate the PDF: f(x) = (1 / (σ * sqrt(2π))) * e^(-((x - μ)^2) / (2σ^2))
        coefficient = 1 / (self.stddev * (2*self.pi)**.5)
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        return coefficient * (e ** exponent)

    def cdf(self, x):
        """
        Calculate the Cumulative Distribution Function (CDF) for a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The CDF value for x (probability that X <= x).
        """
        # Define constants
        e = 2.7182818285
        pi = 3.1415926536
        
        # Calculate the CDF: F(x) = (1 / 2) * (1 + erf((x - μ) / (σ * sqrt(2))))
        coefficient = 1 / 2
        argument = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / pi ** 0.5) * (argument - (argument ** 3) / 3 + (argument ** 5) / 10 - (argument ** 7) / 42 + (argument ** 9) / 216)
        return coefficient * (1 + erf)
