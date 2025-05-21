#!/usr/bin/env python3
"""forecast_btc"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, LSTM, GRU  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau  # type: ignore
from sklearn.metrics import mean_squared_error
import math

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_preprocessed_data(data_dir='preprocessed_data'):
    """
    Load the preprocessed data from disk
    """
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    X_val = np.load(f"{data_dir}/X_val.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")

    with open(f"{data_dir}/scalers.pkl", 'rb') as f:
        scalers = pickle.load(f)

    # Check for NaNs
    has_nans = (np.isnan(X_train).any() or np.isnan(y_train).any() or
                np.isnan(X_val).any() or np.isnan(y_val).any() or
                np.isnan(X_test).any() or np.isnan(y_test).any())
    if has_nans:
        raise ValueError("NaN values found in preprocessed data")

    return X_train, y_train, X_val, y_val, X_test, y_test, scalers


def create_tf_datasets(X_train, y_train, X_val, y_val, batch_size=64):
    """
    Create TensorFlow datasets for training and validation
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(
        buffer_size=1000).batch(batch_size).prefetch(
        tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


def build_lstm_model(input_shape):
    """
    Build an LSTM model for time series forecasting
    """
    model = Sequential([
        LSTM(
            32,
            return_sequences=False,
            activation='tanh',
            input_shape=input_shape),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss='mse')
    return model


def build_gru_model(input_shape):
    """
    Build a GRU model for time series forecasting
    """
    model = Sequential([
        GRU(32,
            return_sequences=False,
            activation='tanh',
            input_shape=input_shape),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss='mse')
    return model


def train_model(model, train_dataset, val_dataset, epochs=50):
    """
    Train the model with callbacks for early
    stopping and learning rate reduction
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-5
    )

    model_checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test, scalers):
    """
    Evaluate the model on test data and visualize predictions
    """
    predictions = model.predict(X_test)

    # Debug: Check predictions
    print(
        "Predictions min/max/mean:",
        predictions.min(),
        predictions.max(),
        predictions.mean())
    print("NaNs in predictions:", np.isnan(predictions).sum())

    close_scaler = scalers['Close']
    predictions_actual = close_scaler.inverse_transform(
        predictions.reshape(-1, 1)).flatten()
    actual_values = close_scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()

    # Debug: Check inverse-scaled values
    print(
        "Predictions_actual min/max/mean:",
        predictions_actual.min(),
        predictions_actual.max(),
        predictions_actual.mean())
    print(
        "Actual_values min/max/mean:",
        actual_values.min(),
        actual_values.max(),
        actual_values.mean())
    print("NaNs in predictions_actual:", np.isnan(predictions_actual).sum())
    print("NaNs in actual_values:", np.isnan(actual_values).sum())

    if np.isnan(predictions_actual).any() or np.isnan(actual_values).any():
        raise ValueError("NaN values found in predictions or actual values")

    rmse = math.sqrt(mean_squared_error(actual_values, predictions_actual))
    print(f"Test RMSE: {rmse}")

    mape = np.mean(
        np.abs(
            (actual_values - predictions_actual) / actual_values)) * 100
    print(f"Test MAPE: {mape}%")

    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual BTC Price')
    plt.plot(predictions_actual, label='Predicted BTC Price')
    plt.title('BTC Price Prediction')
    plt.xlabel('Time (hours)')
    plt.ylabel('BTC Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('btc_predictions.png')
    plt.close()

    sample_size = min(100, len(actual_values))
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values[-sample_size:], label='Actual BTC Price')
    plt.plot(predictions_actual[-sample_size:], label='Predicted BTC Price')
    plt.title('BTC Price Prediction (Last 100 Hours)')
    plt.xlabel('Time (hours)')
    plt.ylabel('BTC Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('btc_predictions_sample.png')
    plt.close()

    return rmse, mape, actual_values, predictions_actual
