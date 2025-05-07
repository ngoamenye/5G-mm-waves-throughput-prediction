#!/usr/bin/env python3
import os

# ✅ Désactiver la recherche de GPU pour éviter les erreurs DLL
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib

import mlflow
import mlflow.tensorflow

from preprocess import load_data, build_sequences, scale_features
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks

# ─── CONFIGURATION ────────────────────────────────────────────────
DATA_PATH = "data/mm-5G-enriched.csv"
SEQ_LEN = 10
MODEL_DIR = "models"
OUTPUTS_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─── MLflow SETUP ─────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("5G-throughput-experiments")
mlflow.tensorflow.autolog()

# ─── RUN TRAINING ─────────────────────────────────────────────────
with mlflow.start_run():
    print("🔹 Loading data...")
    df = load_data(DATA_PATH)
    X, y_raw, y_smooth, runs, feature_names = build_sequences(df, SEQ_LEN)
    X, scaler = scale_features(X)

    # Save pre-processing artifacts
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.gz"))
    mlflow.log_artifact(
        os.path.join(MODEL_DIR, "feature_names.pkl"), artifact_path="preprocessing"
    )
    mlflow.log_artifact(
        os.path.join(MODEL_DIR, "scaler.gz"), artifact_path="preprocessing"
    )

    print("🔹 Splitting data...")
    (
        X_train,
        X_val,
        y_raw_train,
        y_raw_val,
        y_smooth_train,
        y_smooth_val,
        runs_train,
        runs_val,
    ) = train_test_split(
        X, y_raw, y_smooth, runs, test_size=0.2, random_state=42, shuffle=False
    )

    print("🔹 Building model...")
    nf = X.shape[2]
    inp = layers.Input(shape=(SEQ_LEN, nf))
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64)(x)
    raw_out = layers.Dense(1, name="raw")(x)
    smooth_out = layers.Dense(1, name="smooth")(x)
    model = models.Model(inputs=inp, outputs=[raw_out, smooth_out])
    model.compile(optimizer="adam", loss="mse")

    print("🔹 Training model...")
    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        [y_raw_train, y_smooth_train],
        validation_data=(X_val, [y_raw_val, y_smooth_val]),
        epochs=5,
        batch_size=32,
        callbacks=[es],
    )

    print("🔹 Logging model to MLflow...")
    mlflow.keras.log_model(model, artifact_path="throughput_model")

    print("🔹 Saving model weights...")
    model.save(os.path.join(MODEL_DIR, "throughput_model.keras"))
    model.save_weights(os.path.join(MODEL_DIR, "throughput_weights.weights.h5"))

    print("🔹 Predicting on validation set...")
    pred_raw, pred_smooth = model.predict(X_val)
    pred_final = np.where(
        np.abs(pred_raw.squeeze() - y_raw_val)
        < np.abs(pred_smooth.squeeze() - y_raw_val),
        pred_raw.squeeze(),
        pred_smooth.squeeze(),
    )
    val_df = pd.DataFrame(
        {
            "run_num": runs_val,
            "true_raw": y_raw_val,
            "pred_raw": pred_raw.squeeze(),
            "pred_smooth": pred_smooth.squeeze(),
            "pred_final": pred_final,
        }
    )
    file_val = os.path.join(OUTPUTS_DIR, "validation_predictions.csv")
    val_df.to_csv(file_val, index=False)
    mlflow.log_artifact(file_val, artifact_path="validation")

print("✅ Training completed. Check MLflow for logs.")
