#!/usr/bin/env python3
""" Exponential distribution """


class Exponential:
    def __init__(self, data=None, lambtha=1.0):
        """
        Initialize an Exponential distribution.

        Parameters:
        - data (list, optional): List of data to estimate the distribution.
        - lambtha (float, optional): Expected number
        of occurrences in a given time frame.

        Sets the instance attribute:
        - lambtha (float): The rate parameter of the Exponential distribution.

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
            # Calculate lambtha as the inverse of the mean of the data
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """
        Calculate the Probability Density Function
        (PDF) for a given time period.

        Parameters:
        - x (float): Time period.

        Returns:
        - float: The PDF value for x (probability density at time x).

        Notes:
        - If x is out of range (e.g., negative), returns 0.
        """
        # If x is negative, return 0
        if x < 0:
            return 0

        # Define e (base of natural logarithm)
        e = 2.7182818285

        # Calculate the PDF: f(x) = λ * e^(-λx)
        # where λ is self.lambtha
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculate the Cumulative Distribution
        Function (CDF) for a given time period.

        Parameters:
        - x (float): Time period.

        Returns:
        - float: The CDF value for x
        (probability of time being less than or equal to x).

        Notes:
        - If x is out of range (e.g., negative), returns 0.
        """
        # If x is negative, return 0
        if x < 0:
            return 0

        # Define e (base of natural logarithm)
        e = 2.7182818285

        # Calculate the CDF: F(x) = 1 - e^(-λx)
        # where λ is self.lambtha
        return 1 - (e ** (-self.lambtha * x))
