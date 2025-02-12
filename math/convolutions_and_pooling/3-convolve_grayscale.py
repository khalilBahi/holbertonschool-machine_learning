#!/usr/bin/env python3
"""Task: 3. Strided Convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding and stride.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).
        padding (tuple): (ph, pw) padding along the height and width dimensions
        stride (tuple): (sh, sw) the stride for the height and width dimensions

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'same':
        pad_h = int(((h - 1) * sh + kh - h) / 2) + 1
        pad_w = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding
    image_pad = np.pad(images, pad_width=((0, 0),
                                          (pad_h, pad_h), (pad_w, pad_w)),
                       mode='constant')
    H = int((h + 2 * pad_h - kh) / sh + 1)
    W = int((w + 2 * pad_w - kw) / sw + 1)
    conv_images = np.zeros((m, H, W))
    for i in range(H):
        for j in range(W):
            conv_images[:, i, j] = np.sum(
                image_pad[:, i * sh:i * sh + kh, j * sw:j * sw + kw
                          ] * kernel, axis=(1, 2))
    return conv_images
