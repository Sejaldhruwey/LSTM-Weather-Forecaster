
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


FEATURES = ["T", "rh", "p", "wv", "SWDR"]
TARGET = ["T"]


def load_and_clean_data(file_path):
    """Load, validate, and clean weather data."""

    df = pd.read_csv(file_path)

    required_columns = ["date"] + FEATURES
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset="date", keep="first")
    df = df.set_index("date")

    df = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    # Replace invalid sentinel values.
    df = df.replace(-9999, np.nan)

    # Fill missing values using time-based interpolation.
    df = df.interpolate(method="time").ffill()

    if df.isnull().any().any():
        raise ValueError("Dataset still contains missing values.")

    return df


def prepare_train_test_data(
    df,
    train_ratio=0.8,
    sequence_length=144,
    forecast_horizon=144
):
    """
    Split data chronologically, fit scalers only on training data,
    and generate sequences for training and testing.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if len(df) <= sequence_length + forecast_horizon:
        raise ValueError("Not enough observations to create sequences.")

    split_index = int(len(df) * train_ratio)

    # Fit scalers only on the training period.
    train_df = df.iloc[:split_index]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_x.fit(train_df[FEATURES])
    scaler_y.fit(train_df[TARGET])

    # Transform the complete dataset using training-fitted scalers.
    scaled_features = scaler_x.transform(df[FEATURES])
    scaled_target = scaler_y.transform(df[TARGET])

    X_train, y_train = [], []
    X_test, y_test = [], []

    # Generate sequences using their target positions.
    first_target = sequence_length + forecast_horizon - 1

    for target_index in range(first_target, len(df)):

        start_index = target_index - sequence_length - forecast_horizon + 1
        end_index = start_index + sequence_length

        X_sequence = scaled_features[start_index:end_index]
        y_value = scaled_target[target_index]

        if target_index < split_index:
            X_train.append(X_sequence)
            y_train.append(y_value)
        else:
            X_test.append(X_sequence)
            y_test.append(y_value)

    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test),
        scaler_x,
        scaler_y
    )
