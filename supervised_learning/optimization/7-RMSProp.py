#!/usr/bin/env python3
"""Task 7: 7. RMSProp"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight (decay rate for the second moment).
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of the variable.
        s (numpy.ndarray): The previous second moment of the variable.

    Returns:
        tuple: The updated variable and the new second moment, respectively.
    """
    # Update the second moment of the gradient
    s = beta2 * s + (1 - beta2) * grad**2

    # Update the variable using the RMSProp formula
    var = var - alpha * (grad / (np.sqrt(s) + epsilon))

    return var, s
