#!/usr/bin/env python3
"""Task 4: 4. Moving Average"""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the exponentially weighted moving
    average of a dataset with bias correction.

    Parameters:
    data (list): The list of data points to calculate the moving average for.
    beta (float): The weight (decay factor) for the
    moving average (0 < beta < 1).

    Returns:
    list: A list containing the moving averages of the data.
    """
    moving_averages = []  # List to store the moving averages
    v = 0  # Initialize the moving average variable

    for i in range(len(data)):
        # Update the moving average
        v = beta * v + (1 - beta) * data[i]
        # Apply bias correction
        v_corrected = v / (1 - beta**(i + 1))
        moving_averages.append(v_corrected)

    return moving_averages
