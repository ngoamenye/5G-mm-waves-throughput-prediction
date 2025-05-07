import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load model and scaler
MODEL_PATH = "models/throughput_model.keras"
SCALER_PATH = "models/scaler.gz"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Please ensure 'throughput_model.keras' is trained and exists in models/."
    )
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("Scaler file not found. Please ensure 'scaler.gz' exists in models/.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(optimizer="adam", loss="mse")
scaler = joblib.load(SCALER_PATH)

# Load enriched dataset
df = pd.read_csv("data/mm-5G-enriched.csv")

# Configuration
SEQ_LEN = 10
encode = {"Stationary": 0, "Walking": 1, "Driving": 2}

# Streamlit UI
st.set_page_config(layout="wide", page_title="5G Throughput Simulator")
st.title("5G Throughput Simulator with Real Dataset Scenarios")

# Sidebar controls
st.sidebar.header("Scenario Selection")
selected_run = st.sidebar.selectbox(
    "Choose a Simulation Run", sorted(df["run_num"].unique())
)

# Filter run from dataset
run_df = df[df["run_num"] == selected_run].reset_index(drop=True)

# Detect feature columns automatically (excluding targets and ID cols)
exclude = [
    "throughput",
    "throughput_smoothed",
    "debit_brut",
    "debit_lisse",
    "debit_class",
    "run_num",
]
features = [col for col in run_df.columns if col not in exclude]

# Map initialization
start_coord = [run_df.iloc[0]["latitude"], run_df.iloc[0]["longitude"]]
m = folium.Map(location=start_coord, zoom_start=16)

# Prediction loop
predictions = []
for i in range(len(run_df) - SEQ_LEN):
    window = run_df.iloc[i : i + SEQ_LEN]
    raw_feats = window[features].to_numpy()
    scaled_feats = scaler.transform(raw_feats).reshape(1, SEQ_LEN, -1)
    pred_raw, pred_smooth = model.predict(scaled_feats, verbose=0)
    pred = (
        float(pred_raw.squeeze())
        if abs(pred_raw - pred_smooth) < abs(pred_smooth - pred_raw)
        else float(pred_smooth.squeeze())
    )
    predictions.append(pred)

    # Add step marker on the map
    step_lat = run_df.iloc[i + SEQ_LEN - 1]["latitude"]
    step_lon = run_df.iloc[i + SEQ_LEN - 1]["longitude"]
    folium.CircleMarker(
        location=[step_lat, step_lon],
        radius=3,
        color="red",
        fill=True,
        fill_opacity=0.7,
    ).add_to(m)

# Show map
st.subheader("Predicted Path on Map")
st_data = st_folium(m, width=800, height=600)

# Display metrics summary
st.subheader("Simulation Summary")
st.markdown(f"**Selected Run:** {selected_run}")
st.markdown(f"**Steps Processed:** {len(predictions)}")
st.line_chart(
    predictions, use_container_width=True, y_label="Predicted Throughput (Mbps)"
)

# Optional: Save scenario results
if st.button("Export Predictions"):
    pred_df = pd.DataFrame(
        {"step": list(range(len(predictions))), "predicted_throughput": predictions}
    )
    os.makedirs("outputs", exist_ok=True)
    pred_df.to_csv(f"outputs/predictions_run_{selected_run}.csv", index=False)
    st.success(f"Predictions for run {selected_run} exported to outputs/")
