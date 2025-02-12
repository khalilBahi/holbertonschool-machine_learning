#!/usr/bin/env python3
"""Task: 6. Pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w, c).
        kernel_shape (tuple): Shape of the pooling kernel (kh, kw).
        stride (tuple): Stride for the pooling (sh, sw).
        mode (str): Pooling mode, either 'max' or 'avg'.

    Returns:
        numpy.ndarray: Pooled images with shape (m, h_pool, w_pool, c).
    """
    # Get dimensions of input images and kernel
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate dimensions of the output
    h_pool = int((h - kh) / sh + 1)
    w_pool = int((w - kw) / sw + 1)
    pool_images = np.zeros((m, h_pool, w_pool, c))

    # Perform the pooling
    for i in range(h_pool):
        for j in range(w_pool):
            image_slide = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pool_images[:, i, j] = np.max(image_slide, axis=(1, 2))
            elif mode == 'avg':
                pool_images[:, i, j] = np.mean(image_slide, axis=(1, 2))

    return pool_images
