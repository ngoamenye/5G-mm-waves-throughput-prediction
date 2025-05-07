import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# 📌 Setup
DATA_PATH = "data/mm-5G-enriched.csv"
TARGET = "throughput"
EXPERIMENT_NAME = "SynergyX-CNN-LSTM"

# 📍 Start MLflow tracking
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    # 🔢 Load and prepare data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for Conv1D/LSTM [samples, timesteps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 🧠 Define model
    model = Sequential([
        Conv1D(64, kernel_size=1, activation='relu', input_shape=(1, X.shape[1])),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 🔁 Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    # ✅ Evaluate
    loss, mae = model.evaluate(X_test, y_test)

    # 🧪 Log params and metrics
    mlflow.log_param("model_type", "CNN + LSTM")
    mlflow.log_metric("loss", loss)
    mlflow.log_metric("mae", mae)

    # 📂 Save model
    model.save("ml_models/throughput_model.keras")
    mlflow.keras.log_model(model, "model")

    # 🔬 Save artifacts
    plot_model(model, to_file="ml_models/model_architecture.png", show_shapes=True)
    mlflow.log_artifact("ml_models/model_architecture.png")

    print(f"✅ Training complete. Loss: {loss}, MAE: {mae}")
