#!/usr/bin/env python3
"""scatter.py: A script to plot a scatter plot
of synthetic data for men's height and weight.

This script generates a scatter plot of 2,000 synthetic data points
representing men's height and weight. The data is sampled from a multivariate
normal distribution with specified mean and covariance. The scatter plot
visualizes the relationship between height and weight.

Functions:
    scatter():
        Generates and displays a scatter plot of
        the synthetic height and weight data.

Modules:
    - numpy: Used for generating random data
    points from a multivariate normal distribution.
    - matplotlib.pyplot: Used for creating static,
    animated, and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Generates and displays a scatter plot of men's height vs. weight.

    This function performs the following:
    1. Samples 2,000 data points from a multivariate normal distribution with:
       - Mean: [69, 0]
       - Covariance: [[15, 8], [8, 15]]
    2. Adds 180 to the weight values to simulate realistic weight data.
    3. Creates a scatter plot with:
       - Magenta-colored points (#FF00FF) at 70% opacity (alpha=0.7).
       - Marker size of 20.
    4. Labels the x-axis as "Height (in)" and the y-axis as "Weight (lbs)".
    5. Sets the title of the plot as "Men's Height vs Weight".
    6. Displays the plot.

    Example:
    >>> scatter()  # Displays the height vs. weight scatter plot.
    """
    mean = [69, 0]  # Mean of the multivariate normal distribution
    cov = [[15, 8], [8, 15]]  # Covariance matrix
    np.random.seed(5)  # Set the seed for reproducibility

    # Generate synthetic height (x) and weight (y) data
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180  # Shift weight values to a realistic range

    plt.figure(figsize=(6.4, 4.8))  # Configure figure size

    # Plot the scatter plot
    plt.scatter(x, y, color='#FF00FF')

    # Add labels and title
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")

    # Display the plot
    plt.show()
