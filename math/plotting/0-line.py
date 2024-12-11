#!/usr/bin/env python3
"""line.py: A script to plot a cubic function using Matplotlib.

This script generates a simple line plot of
the function y = x^3 for x in the range 0 to 10.
The graph is displayed using Matplotlib.

Functions:
    line():
        Generates and displays a plot of the function y = x^3.

Modules:
    - numpy: Used for efficient numerical operations.
    - matplotlib.pyplot: Used for creating static,
      animated, and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Generates and displays a plot of the cubic function y = x^3.

    This function performs the following:
    1. Computes the cubic values for x in the range 0 to 10.
    2. Configures the plot figure size.
    3. Plots the cubic function with a solid red line.
    4. Sets the x-axis range to [0, 10].
    5. Displays the plot.

    Details:
    - The plot uses a solid red line ('r-') to represent the cubic function.
    - The x-axis is restricted to the range [0, 10] for clarity.
    - A legend can be added if a label is provided to the plot.

    Example:
    >>> line()  # This will display the cubic plot.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # Plot the graph
    plt.plot(range(11), y, "r-")  # Solid red line

    # Set x-axis range
    plt.xlim(0, 10)

    # Show the plot
    plt.show()
