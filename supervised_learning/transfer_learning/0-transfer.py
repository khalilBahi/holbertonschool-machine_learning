#!/usr/bin/env python3
"""Task 0. Transfer Knowledge"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import numpy as np


def preprocess_data(X, Y):
    """
    Preprocesses the input data for compatibility with the ResNet50 model.

    Args:
        X (numpy.ndarray): Input image data, shape (m, 32, 32, 3).
        Y (numpy.ndarray): Labels corresponding to the input data, shape (m,).

    Returns:
        X_p (numpy.ndarray): Preprocessed image data.
        Y_p (numpy.ndarray): One-hot encoded labels.
    """
    # Resize images to 224x224 and preprocess for ResNet50
    X_p = tf.image.resize(X, [224, 224])
    X_p = tf.keras.applications.resnet50.preprocess_input(X_p)
    # One-hot encode the labels
    Y_p = to_categorical(Y, num_classes=10)
    return X_p, Y_p


def resize_image(X):
    """
    Resizes input images to 224x224 to be compatible with the ResNet50 model.

    Args:
        X (tensor): Input image data.

    Returns:
        resized_X (tensor): Resized image data.
    """
    return tf.image.resize(X, [224, 224])


def build_model():
    """
    Builds the model using ResNet50 as the base model.

    Returns:
        model (tf.keras.Model): Compiled model.
    """
    # Load ResNet50 with pre-trained weights, excluding the top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(
            224,
            224,
            3))

    # Freeze the base model layers
    base_model.trainable = False

    # Define the input layer
    input = layers.Input(shape=(32, 32, 3))

    # Resize images using a Lambda layer
    resized_input = layers.Lambda(resize_image)(input)

    # Pass the resized input through the base model
    x = base_model(resized_input, training=False)

    # Add custom layers
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(input, output)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model():
    """
    Trains the model on the CIFAR-10 dataset and saves it as cifar10.h5.
    """
    # Load CIFAR-10 data
    (X_train, Y_train), (X_test,
                         Y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Build the model
    model = build_model()

    # Print model summary
    model.summary()

    # Define callbacks
    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='cifar10.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]

    # Train the model
    history = model.fit(X_train_p, Y_train_p,
                        batch_size=256,
                        epochs=20,
                        validation_data=(X_test_p, Y_test_p),
                        callbacks=my_callbacks)


if __name__ == "__main__":
    train_model()
