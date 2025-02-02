#!/usr/bin/env python3
"""Task 7: 7. Early Stopping"""
import tensorflow as tf


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.
    Parameters:
    cost -- the current validation cost of the neural network.
    opt_cost -- the lowest recorded validation cost of the neural network.
    threshold -- the threshold used for early stopping.
    patience -- the patience count used for early stopping.
    count -- the count of how long the threshold has not been met.
    Returns:
    boolean of whether the network should be stopped
    early, followed by the updated count.
    """
    if opt_cost - cost > threshold:
        return False, 0
    else:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count
