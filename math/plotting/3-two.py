#!/usr/bin/env python3
"""
Generate and display a plot of the exponential decay of C-14 and Ra-226.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Generate and display a plot of the exponential decay of C-14 and Ra-226.

    Args:
             None

    Returns:
             None
    """

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y1, "r--", label="C-14")
    plt.plot(x, y2, "g-", label="Ra-226")

    # Labeling and title
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")

    # Set x-axis range
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Display the plot
    plt.legend()
    plt.show()
