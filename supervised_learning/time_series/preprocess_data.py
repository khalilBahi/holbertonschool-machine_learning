#!/usr/bin/env python3
"""preprocess_data"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pickle


def load_and_clean_data(file_path):
    """
    Load and clean the Bitcoin data from CSV files
    """
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    df = df.sort_index()
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    # Filter positive prices (but not volumes yet)
    df = df[df['Close'] > 0]
    return df


def resample_data(df, freq='1h'):
    """
    Resample data to hourly intervals
    """
    resampled = df.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })
    # Forward fill any NaNs introduced by resampling
    resampled = resampled.ffill()
    # Filter out zero or negative volumes
    resampled = resampled[resampled['Volume_(BTC)'] > 0]
    print("NaNs after resampling and filtering:", resampled.isna().sum())
    print("Volume_(BTC) min after filtering:", resampled['Volume_(BTC)'].min())
    return resampled


def prepare_features_and_targets(df, sequence_length=24):
    """
    Prepare features and targets for the RNN model
    """
    selected_features = ['Close', 'Volume_(BTC)', 'Weighted_Price']
    data = df[selected_features]

    # Debug: Check data statistics
    print("Data stats before scaling:")
    print(data.describe())
    print("NaNs in data:", data.isna().sum())

    # Scale features independently
    scalers = {}
    scaled_data = np.zeros_like(data.values)
    for i, feature in enumerate(selected_features):
        scalers[feature] = MinMaxScaler(feature_range=(0, 1))
        scaled_data[:, i] = scalers[feature].fit_transform(
            data[[feature]].values).flatten()

    # Debug: Check scaled data
    print(
        "Scaled data min/max/mean:",
        scaled_data.min(),
        scaled_data.max(),
        scaled_data.mean())
    print("NaNs in scaled data:", np.isnan(scaled_data).sum())

    features, targets = [], []
    for i in range(len(scaled_data) - sequence_length - 1):
        features.append(scaled_data[i:i + sequence_length])
        # Next hour close price
        targets.append(scaled_data[i + sequence_length, 0])

    features = np.array(features)
    targets = np.array(targets)

    # Debug: Check features and targets
    print(
        "Features min/max/mean:",
        features.min(),
        features.max(),
        features.mean())
    print(
        "Targets min/max/mean:",
        targets.min(),
        targets.max(),
        targets.mean())
    print("NaNs in features:", np.isnan(features).sum())
    print("NaNs in targets:", np.isnan(targets).sum())

    # Save the close price scaler for inverse scaling
    with open('preprocessed_data/close_scaler.pkl', 'wb') as f:
        pickle.dump(scalers['Close'], f)

    return features, targets, scalers
