#!/usr/bin/env python3
"""
Module for image hue adjustment operations
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image

    Args:
        image: A 3D tf.Tensor containing the image to change
        delta: The amount the hue should change

    Returns:
        The altered image as a tf.Tensor
    """
    return tf.image.adjust_hue(image, delta)
