#!/usr/bin/env python3
"""
Binomial distribution
"""
import math


class Binomial:
    """class Binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution parameters.

        If data is provided, estimates the distribution parameters (n, p)
        based on the sample data.

        Parameters:
        data (list, optional):
            A dataset used to estimate the distribution parameters.
            Defaults to None.
        n (int, optional):
            The number of Bernoulli trials. Defaults to 1.
        p (float, optional):
            The probability of success in each trial. Defaults to 0.5.

        Raises:
        TypeError: If data is not a list.
        ValueError: If data contains fewer than two values.
        ValueError: If n is not a positive integer.
        ValueError: If p is not in the range (0, 1).
        """
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            new_data = [(x - mean) ** 2 for x in data]
            variance = sum(new_data) / len(data)
            p = 1 - variance / mean
            if ((mean / p) - (mean // p)) >= 0.5:
                self.n = 1 + int(mean / p)
            else:
                self.n = int(mean / p)
            self.p = float(mean / self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
            self.n = int(n)

    def pmf(self, k):
        """
        Calculate the Probability Mass Function (PMF)
        for a given number of successes.

        Parameters:
        - k (int or float): Number of successes.

        Returns:
        - float: The PMF value for k (probability of exactly k successes).

        Notes:
        - If k is not an integer, it is converted to an integer.
        - If k is out of range (e.g., negative or greater than n), returns 0.
        """
        # Convert k to an integer if it is not
        k = int(k)

        # If k is out of range (k < 0 or k > n), return 0
        if k < 0 or k > self.n:
            return 0

        # Calculate the binomial coefficient: C(n, k) = n! / (k! * (n-k)!)
        def binomial_coefficient(n, k):
            # Optimize by using the smaller value for the loop
            if k > n - k:
                k = n - k
            coefficient = 1
            for i in range(k):
                coefficient *= n - i
                coefficient //= i + 1
            return coefficient

        # Calculate the PMF: P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
        coefficient = binomial_coefficient(self.n, k)
        success_prob = self.p**k
        failure_prob = (1 - self.p) ** (self.n - k)
        return coefficient * success_prob * failure_prob

    def cdf(self, k):
        """
        Calculate the Cumulative Distribution Function
        CDF) for a given number of successes.

        Parameters:
        - k (int or float): Number of successes.

        Returns:
        - float: The CDF value for k (probability of at most k successes).

        Notes:
        - If k is not an integer, it is converted to an integer.
        - If k is out of range (e.g., negative), returns 0.
        - If k is greater than or equal to n, returns 1.
        - Uses the pmf method to compute the CDF
        as the sum of PMF values from 0 to k.
        """
        # Convert k to an integer if it is not
        k = int(k)

        # If k is negative, return 0
        if k < 0:
            return 0

        # If k is greater than or equal to n,
        # return 1 (since CDF is cumulative up to n)
        if k >= self.n:
            return 1.0

        # Calculate the CDF: P(X <= k) = sum(P(X = i) for i from 0 to k)
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
