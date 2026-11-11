
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from preprocess import load_and_clean_data, prepare_train_test_data


def build_lstm_model(input_shape):
    """Build the LSTM forecasting model."""

    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mean_squared_error"
    )

    return model


def main():

    # Load cleaned weather data.
    df = load_and_clean_data("weather_cleaned_v2.csv")

    # Prepare training and testing sequences.
    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler_x,
        scaler_y
    ) = prepare_train_test_data(df)

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Build the model.
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    
    # Train the model.
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=256,
        shuffle=False,
        verbose=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )
        ]
        
    )
    
    
    

    # Plot training and validation loss.
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Generate predictions.
    predictions = model.predict(X_test)

    # Convert scaled values back to Celsius.
    predictions = scaler_y.inverse_transform(predictions)
    actual_values = scaler_y.inverse_transform(y_test)

    # Evaluate model performance.
    r2 = r2_score(actual_values, predictions)
    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))

    print("\nModel Performance")
    print("-------------------------")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f} °C")
    print(f"RMSE: {rmse:.4f} °C")

    # Save model and scalers.
    model.save("weather_model.keras")
    joblib.dump(scaler_x, "scaler_x.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

    print("\nModel and scalers saved successfully.")


if __name__ == "__main__":
    main()
