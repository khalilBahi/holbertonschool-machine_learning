#!/usr/bin/env python3
"""Task 5. Multiple Kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple kernels.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w, c).
        kernels (numpy.ndarray): Convolution kernels with shape (kh, kw, kc, nc
        padding (str/tuple): 'same', 'valid', or a tuple of (ph, pw).
        stride (tuple): Stride for the convolution (sh, sw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, h_conv, w_conv, nc).
    """
    # Get dimensions of input images and kernels
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    # Determine padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    else:
        ph, pw = padding

    # Pad the images
    image_pad = np.pad(images, pad_width=((0, 0),
                                          (ph, ph), (pw, pw), (0, 0)),
                       mode='constant')

    # Calculate dimensions of the output
    h_conv = int(((h + 2 * ph - kh) / sh) + 1)
    w_conv = int(((w + 2 * pw - kw) / sw) + 1)
    conv_images = np.zeros((m, h_conv, w_conv, nc))

    # Perform the convolution
    for i in range(h_conv):
        for j in range(w_conv):
            for z in range(nc):
                image_slide = image_pad[:, i * sh:i *
                                        sh + kh, j * sw:j * sw + kw, :]
                conv_images[:, i, j, z] = np.sum(
                    image_slide * kernels[:, :, :, z], axis=(1, 2, 3))

    return conv_images
