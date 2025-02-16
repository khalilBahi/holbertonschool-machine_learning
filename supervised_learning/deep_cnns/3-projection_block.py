#!/usr/bin/env python3
"""Task 3. Projection Block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep
    Residual Learning for Image Recognition (2015).

    Arguments:
    A_prev -- output from the previous layer (tensor of shape (H, W, C))
    filters -- list or tuple containing F11, F3, F12:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1
        convolution and the shortcut connection
    s -- stride of the first convolution in both the main
    path and the shortcut connection (default is 2)

    Returns:
    activated_output -- the activated output of the
    projection block (tensor of shape (H/s, W/s, F12))
    """
    F11, F3, F12 = filters

    # He normal initializer with seed=0
    initializer = K.initializers.HeNormal(seed=0)

    # Main path
    # First 1x1 convolution
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        kernel_initializer=initializer,
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation("relu")(bn1)

    # Second 3x3 convolution
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer,
    )(relu1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation("relu")(bn2)

    # Third 1x1 convolution
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=initializer,
    )(relu2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Shortcut path
    conv_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        kernel_initializer=initializer,
    )(A_prev)
    bn_shortcut = K.layers.BatchNormalization(axis=3)(conv_shortcut)

    # Add main path and shortcut
    add = K.layers.Add()([bn3, bn_shortcut])
    activated_output = K.layers.Activation("relu")(add)

    return activated_output
