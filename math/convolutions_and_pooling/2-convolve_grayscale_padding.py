#!/usr/bin/env python3
"""Task: 2. Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).
        padding (tuple): (ph, pw) padding for the height and width.

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    image_pad = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                       mode='constant')
    H = int(h + 2 * ph - kh + 1)
    W = int(w + 2 * pw - kw + 1)
    conv_images = np.zeros((m, H, W))
    for i in range(H):
        for j in range(W):
            conv_images[:, i, j] = np.sum(
                image_pad[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))
    return conv_images
