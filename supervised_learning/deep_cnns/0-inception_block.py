#!/usr/bin/env python3
"""Task 0. Inception Block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    Going Deeper with Convolutions (2014).

    Arguments:
    A_prev -- output from the previous layer
    filters -- tuple or list containing F1, F3R, F3,F5R, F5, FPP:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution
        before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution
        before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution
        after the max pooling

    Returns:
    Concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                            padding='same', activation='relu',
                            kernel_initializer=init)(A_prev)

    conv3R = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)
    conv3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                            padding='same', activation='relu',
                            kernel_initializer=init)(conv3R)

    conv5R = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)
    conv5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5),
                            padding='same', activation='relu',
                            kernel_initializer=init)(conv5R)

    pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    convPP = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                             padding='same', activation='relu',
                             kernel_initializer=init)(pool)

    return K.layers.concatenate([conv1, conv3, conv5, convPP])
