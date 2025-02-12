#!/usr/bin/env python3
"""Task 1. Pooling Forward Propagation"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function to forward propagate over a pooling layer.

    Parameters:

        A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev),
        output of the previous layer
        W -- numpy.ndarray of shape (kh, kw, c_prev, c_new),
        kernels for the convolution
        b -- numpy.ndarray of shape (1, 1, 1, c_new), biases
        applied to the convolution
        activation -- activation function to be applied
        padding -- string, either "same" or "valid",
        indicating the type of padding used
        stride -- tuple of (sh, sw), strides for the convolution

    Returns:
            (conv) output of the pooling layer
    """

    # Retrieve the dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieve the dimensions from kernel_shape
    (kh, kw) = kernel_shape

    # Retrieve the values of the stride
    sh, sw = stride

    # Compute the dimensions of the CONV output volume
    c_h = int((h_prev - kh) / sh) + 1
    c_w = int((w_prev - kw) / sw) + 1

    # Initialize the output volume conv (Z) with zeros
    conv = np.zeros((m, c_h, c_w, c_prev))

    # Loop over the vertical_ax, then horizontal_ax, then over channel
    for x in range(c_h):
        for y in range(c_w):
            # pooling implementation
            if mode == 'max':
                conv[:, x, y] = (np.max(A_prev[:,
                                        x * sh:((x * sh) + kh),
                                        y * sw:((y * sw) + kw)],
                                        axis=(1, 2)))
            elif mode == 'avg':
                conv[:, x, y] = (np.mean(A_prev[:,
                                         x * sh:((x * sh) + kh),
                                         y * sw:((y * sw) + kw)],
                                         axis=(1, 2)))
    return conv
