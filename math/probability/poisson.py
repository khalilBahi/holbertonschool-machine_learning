#!/usr/bin/env python3
"""
Poisson distribution
"""


def factorial(n):
    """
    Calculates the factorial of a given number.

    Parameters:
    n (int):
    The number for which the factorial is to be computed.

    Returns:
    int: The factorial of `n`.

    Notes:
    - The factorial of 0 is defined as 1.
    - The function uses an iterative approach.
    """
    if n == 0:
        return 1

    fact = 1

    for i in range(1, n + 1):
        fact = fact * i
    return fact


class Poisson:
    """class Poisson"""

    def __init__(self, data=None, lambtha=1.0):
        """
        Initialize a Poisson distribution.

        Parameters:
        - data (list, optional): List of data to estimate the distribution.
        - lambtha (float, optional): Expected number of
        occurrences in a given time frame.

        Sets the instance attribute:
        - lambtha (float): The rate parameter of the Poisson distribution.

        Raises:
        - ValueError: If lambtha is not positive or if
        data does not contain multiple values.
        - TypeError: If data is not a list.
        """
        # Case 1: If data is not given (i.e., data is None)
        if data is None:
            # Check if lambtha is positive
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            # Save lambtha as a float
            self.lambtha = float(lambtha)
        else:
            # Case 2: If data is given
            # Check if data is a list
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            # Check if data has at least two data points
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha as the mean of the data
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculate the Probability Mass Function
        (PMF) for a given number of successes.

        Parameters:
        - k (int or float): Number of successes.

        Returns:
        - float: The PMF value for k
        (probability of observing exactly k successes).

        Notes:
        - If k is not an integer, it is converted to an integer.
        - If k is out of range (e.g., negative), returns 0.
        """
        # Convert k to an integer if it is not
        k = int(k)
        e = 2.7182818285

        # If k is negative, return 0 (out of range for Poisson distribution)
        if k < 0:
            return 0

        # Calculate the PMF: P(X = k) = (e^(-λ) * λ^k) / k!
        # where λ is self.lambtha
        numerator = e ** (-self.lambtha) * (self.lambtha**k)
        denominator = factorial(k)
        return numerator / denominator

    def cdf(self, k):
        """
        Calculate the Cumulative Distribution
        Function (CDF) for a given number of successes.

        Parameters:
        - k (int or float): Number of successes.

        Returns:
        - float: The CDF value for k
        (probability of observing k or fewer successes).

        Notes:
        - If k is not an integer, it is converted to an integer.
        - If k is out of range (e.g., negative), returns 0.
        """
        # Convert k to an integer if it is not
        k = int(k)
        e = 2.7182818285

        # If k is negative, return 0 (out of range for Poisson distribution)
        if k < 0:
            return 0

        # where λ is self.lambtha
        cdf_value = 0
        for i in range(k + 1):
            numerator = e ** (-self.lambtha) * (self.lambtha**i)
            denominator = factorial(i)
            cdf_value += numerator / denominator

        return cdf_value
