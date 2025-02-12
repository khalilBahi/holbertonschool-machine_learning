#!/usr/bin/env python3
"""Task: 0. Valid Convolution"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, h - kh + 1, w - kw + 1).
    """
    # Get the dimensions of the images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape
    H = int(h - kh + 1)
    W = int(w - kw + 1)
    conv_images = np.zeros((m, H, W))

    # Perform the convolution
    for i in range(H):
        for j in range(W):
            conv_images[:, i, j] = np.sum(
                images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return conv_images
