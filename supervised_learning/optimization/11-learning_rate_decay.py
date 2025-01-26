#!/usr/bin/env python3
"""Task 11: 11. Learning Rate Decay"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used
        to determine the rate at which alpha will decay.
        global_step (int): The number of passes
        of gradient descent that have elapsed.
        decay_step (int): The number of passes
        of gradient descent that should occur before alpha is decayed further.

    Returns:
        float: The updated value for alpha.
    """
    # Calculate the number of decay steps that have occurred
    step = np.floor(global_step / decay_step)

    # Update the learning rate using inverse time decay
    decayed_learning_rate = alpha / (1 + decay_rate * step)

    return decayed_learning_rate
