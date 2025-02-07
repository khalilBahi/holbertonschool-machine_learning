#!/usr/bin/env python3
"""change_scale.py: A script to plot the exponential
decay of Carbon-14 with a logarithmic y-axis.

This script generates a plot of the exponential decay
of Carbon-14 (C-14) using the formula y = exp((r / t) * x),
where y represents the fraction remaining, r is the decay
constant, t is the half-life of Carbon-14 (5730 years),
and x is the time in years. The y-axis is scaled
logarithmically to better visualize the decay over time.

Functions:
    change_scale():
        Generates and displays a plot of the exponential
        decay of Carbon-14 with a logarithmic y-axis.

Modules:
    - numpy: Used for efficient numerical operations.
    - matplotlib.pyplot: Used for creating static,
    animated, and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Generates and displays a plot of the exponential decay
    of Carbon-14 (C-14) with a logarithmic y-axis.

    This function performs the following:
    1. Defines the time range from 0 to 28,650 years, with
    intervals of 5730 years (the half-life of C-14).
    2. Calculates the fraction over time using its half-life.
    3. Plots the exponential decay curve with a blue solid line.
    4. Sets the y-axis to a logarithmic scale to better visualize the decay.
    5. Adds axis labels, a title, and displays the plot.

    Details:
    - The plot shows the exponential decay of Carbon-14 (C-14) over time.
    - The x-axis represents time (in years), and the y-axis represents
    the fraction remaining.
    - A logarithmic scale is applied to the y-axis to enhance
    the visibility of the decay.
    - The x-axis is restricted to the range [0, 28,650] years,
    which represents multiple half-lives of C-14.

    Example:
    >>> change_scale()  # This will display the exponential decay
    plot with a logarithmic y-axis.
    """

    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y)

    # Labeling the x-axis and y-axis
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')

    # Adding the title to the plot
    plt.title('Exponential Decay of C-14')

    # Set logarithmic scaling for the y-axis
    plt.yscale('log')

    # Set the range for the x-axis from 0 to 28,650 years
    plt.xlim(0, 28650)

    # Show the plot
    plt.show()
