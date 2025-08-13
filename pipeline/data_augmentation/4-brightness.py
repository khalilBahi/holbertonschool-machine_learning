#!/usr/bin/env python3
"""
Module for image brightness adjustment operations
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    Args:
        image: A 3D tf.Tensor containing the image to change
        max_delta: The maximum amount the image should be brightened (or darkened)

    Returns:
        The altered image as a tf.Tensor
    """
    return tf.image.random_brightness(image, max_delta)
