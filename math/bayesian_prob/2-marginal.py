#!/usr/bin/env python3
"""0. Likelihood"""
import numpy as np


def n_choose_x(n, x):
    """
    Computes the binomial coefficient (n choose x).

    Args:
        n (int): The total number of trials.
        x (int): The number of successful trials.

    Returns:
        float: The computed binomial coefficient.
    Raises:
        ValueError: If n or x is negative or non-integer.
    """
    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    nx_fact = np.math.factorial(n - x)
    return n_fact / (x_fact * nx_fact)


def likelihood(x, n, P):
    """
    Calculate the likelihood of observing x
    successes in n trials given probabilities P.

    Args:
        x: Number of patients with severe side effects
        n: Total number of patients
        P: 1D numpy.ndarray of hypothetical probabilities

    Returns:
        1D numpy.ndarray of likelihoods for each probability in P
    """
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    # Check if x is not greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in P are in [0, 1]
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coeff = n_choose_x(n, x)
    success_rate = pow(P, x)
    failure_rate = pow(1 - P, n - x)

    likelihoods = binomial_coeff * success_rate * failure_rate
    return likelihoods


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining x
    successes in n trials with prior beliefs.

    Args:
        x: Number of patients with severe side effects
        n: Total number of patients
        P: 1D numpy.ndarray of hypothetical probabilities
        Pr: 1D numpy.ndarray of prior beliefs of P

    Returns:
        1D numpy.ndarray containing the intersection for each probability in P
    """
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    # Check if x is not greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if Pr is a numpy.ndarray with same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Check if all values in P are in [0, 1]
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Check if all values in Pr are in [0, 1]
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Check if Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate likelihood using the provided likelihood function
    likelihoods = likelihood(x, n, P)

    # Calculate intersection by multiplying likelihood with prior beliefs
    return likelihoods * Pr


def marginal(x, n, P, Pr):
    """
    Calculate the marginal probability of obtaining x successes in n trials.

    Args:
        x: Number of patients with severe side effects
        n: Total number of patients
        P: 1D numpy.ndarray of hypothetical probabilities
        Pr: 1D numpy.ndarray of prior beliefs of P

    Returns:
        float: The marginal probability of obtaining x and n
    """
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    # Check if x is not greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if Pr is a numpy.ndarray with same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Check if all values in P are in [0, 1]
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Check if all values in Pr are in [0, 1]
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Check if Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    marginal_prob = np.sum(intersection(x, n, P, Pr))

    return marginal_prob
