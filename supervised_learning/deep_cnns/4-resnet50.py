#!/usr/bin/env python3
"""Task 4: ResNet-50"""
from tensorflow import keras as K

identity_block = __import__("2-identity_block").identity_block
projection_block = __import__("3-projection_block").projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep Residual Learning
    for Image Recognition (2015).

    Returns:
    model -- a Keras Model
    """
    input_layer = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(
        filters=64, kernel_size=7, padding="same",
        strides=2, kernel_initializer=init
    )(input_layer)

    batch1 = K.layers.BatchNormalization()(conv1)

    relu1 = K.layers.Activation("relu")(batch1)

    pool_1 = K.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(relu1)

    # first projection block
    projection1 = projection_block(pool_1, [64, 64, 256], 1)

    # first identity blocks
    identity1 = identity_block(projection1, [64, 64, 256])
    identity2 = identity_block(identity1, [64, 64, 256])

    # second projection block
    projection2 = projection_block(identity2, [128, 128, 512])

    # second identity blocks
    identity3 = identity_block(projection2, [128, 128, 512])
    identity4 = identity_block(identity3, [128, 128, 512])
    identity5 = identity_block(identity4, [128, 128, 512])

    # third projection block
    projection3 = projection_block(identity5, [256, 256, 1024])

    # third identity blocks
    identity6 = identity_block(projection3, [256, 256, 1024])
    identity7 = identity_block(identity6, [256, 256, 1024])
    identity8 = identity_block(identity7, [256, 256, 1024])
    identity9 = identity_block(identity8, [256, 256, 1024])
    identity10 = identity_block(identity9, [256, 256, 1024])

    # fourth projection block
    projection4 = projection_block(identity10, [512, 512, 2048])

    # fourth identity blocks
    identity11 = identity_block(projection4, [512, 512, 2048])
    identity12 = identity_block(identity11, [512, 512, 2048])

    # average pool
    avg_pool = K.layers.AveragePooling2D(
        pool_size=7, padding="same")(identity12)

    dense = K.layers.Dense(1000, activation="softmax",
                           kernel_initializer=init)(
        avg_pool
    )

    model = K.models.Model(inputs=input_layer, outputs=dense)

    return model
