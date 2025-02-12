#!/usr/bin/env python3
"""Task: 1. Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images.
    """
    # Get the dimensions of the images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the padding for height and width
    pad_h = int((kh - 1) / 2)
    pad_w = int((kw - 1) / 2)

    # Adjust padding if the kernel dimensions are even
    if kh % 2 == 0:
        pad_h = int(kh / 2)
    if kw % 2 == 0:
        pad_w = int(kw / 2)

    # Pad the images with zeros
    image_pad = np.pad(images, pad_width=((0, 0),
                                          (pad_h, pad_h), (pad_w, pad_w)),
                       mode='constant')

    # Initialize the output array for convolved images
    conv_images = np.zeros((m, h, w))

    # Perform the convolution
    for i in range(h):
        for j in range(w):
            conv_images[:, i, j] = np.sum(
                image_pad[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return conv_images
