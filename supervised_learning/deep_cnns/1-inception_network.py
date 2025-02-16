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
    input_layer = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    # Initial layers
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init)(input_layer)
    conv1 = K.layers.BatchNormalization()(conv1)
    conv1 = K.layers.Activation('relu')(conv1)

    max_pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    conv2 = K.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init)(max_pool1)
    conv2 = K.layers.BatchNormalization()(conv2)
    conv2 = K.layers.Activation('relu')(conv2)

    conv3 = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init)(conv2)
    conv3 = K.layers.BatchNormalization()(conv3)
    conv3 = K.layers.Activation('relu')(conv3)

    max_pool2 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(conv3)

    # Inception blocks
    inception3a = inception_block(
        max_pool2, [64, 96, 128, 16, 32, 32])  # Inception 3a
    inception3b = inception_block(
        inception3a, [
            128, 128, 192, 32, 96, 64])  # Inception 3b
    max_pool3 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(inception3b)

    inception4a = inception_block(
        max_pool3, [192, 96, 208, 16, 48, 64])  # Inception 4a
    inception4b = inception_block(
        inception4a, [
            160, 112, 224, 24, 64, 64])  # Inception 4b
    inception4c = inception_block(
        inception4b, [
            128, 128, 256, 24, 64, 64])  # Inception 4c
    inception4d = inception_block(
        inception4c, [
            112, 144, 288, 32, 64, 64])  # Inception 4d
    inception4e = inception_block(
        inception4d, [
            256, 160, 320, 32, 128, 128])  # Inception 4e
    max_pool4 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(inception4e)

    inception5a = inception_block(
        max_pool4, [256, 160, 320, 32, 128, 128])  # Inception 5a
    inception5b = inception_block(
        inception5a, [
            384, 192, 384, 48, 128, 128])  # Inception 5b

    # Final layers
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1), padding='valid')(inception5b)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)
    flatten = K.layers.Flatten()(dropout)
    dense = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init)(flatten)

    model = K.models.Model(inputs=input_layer, outputs=dense)
    return model
