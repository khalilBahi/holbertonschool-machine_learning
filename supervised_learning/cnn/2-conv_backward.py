#!/usr/bin/env python3
"""Task 2. Convolutional Back Propagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over a convolutional layer
    Arguments:
        dZ -- numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output
        A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W -- numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
            kh is the filter height
            kw is the filter width
        b -- numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
        padding -- string that is either same or valid, indicating the type of
        padding used
        stride -- tuple of (sh, sw) containing the strides for the convolution
            sh is the stride for the height
            sw is the stride for the width
    Returns:
        dA_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev)containing
        the partial derivatives with respect to the previous layer
        dW -- numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        partial derivatives with respect to the kernels
        db -- numpy.ndarray of shape (1, 1, 1, c_new) containing the partial
        derivatives with respect to the biases
    """
    # Retrieve the dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieve dimensions from dZ.shape
    (m, h_new, w_new, c_new) = dZ.shape

    # Retrieve the dimensions from A_prev shape
    (kh, kw, c_prev, c_new) = W.shape

    # Retrieve the values of the stride
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    if padding == 'valid':
        pw = 0
        ph = 0

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Pad with zeros all images of the dataset
    A_prev_pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw),
                                           (0, 0)), mode='constant')
    dA_prev_pad = np.pad(dA_prev, pad_width=((0, 0), (ph, ph), (pw, pw),
                                             (0, 0)), mode='constant')

    # Loop over the vertical_ax, then horizontal_ax, then over channel
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Find the corners of the current slice
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    # Use corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[v_start:v_end, h_start:h_end]

                    # update gradients for the window filter param
                    da_prev_pad[v_start:v_end,
                                h_start:h_end] += \
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == 'same':
            # set the ith training example dA_prev to unppaded da_prev_pad
            dA_prev[i, :, :, :] += da_prev_pad[ph:-ph, pw:-pw, :]
        if padding == 'valid':
            dA_prev[i, :, :, :] += da_prev_pad

    return dA_prev, dW, db
