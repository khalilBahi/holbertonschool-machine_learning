#!/usr/bin/env python3
"""Task 1: Inception Network"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described
    in Going Deeper with Convolutions (2014).

    Returns:
    model -- a Keras Model instance
    """
    # implement He et. al initialization for the layers weights
    init = K.initializers.he_normal(seed=None)

    input_layer = K.Input(shape=(224, 224, 3))

    # Conv 7x7 + 2(S)
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            activation='relu',
                            kernel_initializer=init,
                            )(input_layer)

    # MaxPool 3x3 + 2(S)
    max_pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2))(conv1)

    # Conv 1x1 1(V)
    conv2 = K.layers.Conv2D(filters=64,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=init,
                            )(max_pool1)

    # Conv 3x3 1(S)
    conv3 = K.layers.Conv2D(filters=192,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=init,
                            )(conv2)

    # Max pooling layer with kernels of shape 3x3 with 2x2 strides
    max_pool2 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2))(conv3)

    # Inception blocks
    inception3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])

    # Max pooling layer with kernels of shape 3x3 with 2x2 strides
    max_pool3 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2))(inception3b)

    inception4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])

    # Max pooling layer with kernels of shape 3x3 with 2x2 strides
    max_pool4 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2))(inception4e)

    inception5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])

    # Avg pooling layer with kernels of shape 7x7
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(inception5b)

    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    # Fully connected softmax output layer with 1000 nodes
    dense = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=init,
                           )(dropout)

    model = K.models.Model(inputs=input_layer, outputs=dense)

    return model
