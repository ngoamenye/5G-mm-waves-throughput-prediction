from dotenv import load_dotenv

load_dotenv()

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import tensorflow as tf
from tensorflow.keras import layers, models

# ─── Configuration ─────────────────────────────────────────────────────────────
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")

MODEL_DIR = "models"
WEIGHTS_H5 = os.path.join(MODEL_DIR, "throughput_weights.weights.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.gz")

SEQ_LEN = 10


def build_cnn_lstm_model(seq_len, nf):
    inp = layers.Input(shape=(seq_len, nf))
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64)(x)
    raw_out = layers.Dense(1, name="raw")(x)
    smooth_out = layers.Dense(1, name="smooth")(x)
    return models.Model(inputs=inp, outputs=[raw_out, smooth_out])


# Load scaler & model weights
scaler = joblib.load(SCALER_PATH)
NF_FEATURES = scaler.scale_.shape[0]

model = build_cnn_lstm_model(SEQ_LEN, NF_FEATURES)
try:
    model.load_weights(WEIGHTS_H5)
except Exception:
    st.error(
        "Model weights not found. Please run train_tf_model.py to generate throughput_weights.weights.h5"
    )
    st.stop()

encode = {"Driving": 2, "Walking": 1, "Stationary": 0}

# Streamlit page config
st.set_page_config(layout="wide", page_title="5G Throughput 3D Simulator")
st.title("🚀 5G Throughput 3D Simulator")

# Scenario selection
scenario = st.sidebar.selectbox(
    "Select Scenario", ["Urban Drive → Airport Walk", "Custom"]
)
if scenario == "Urban Drive → Airport Walk":
    mode = "Driving"
    speed = 60
    antenna_df = pd.DataFrame(
        [
            {"lat": 40.7561, "lon": -73.9903, "coverage": 300, "orientation": 45},
            {"lat": 40.7580, "lon": -73.9855, "coverage": 250, "orientation": 135},
            {"lat": 40.7595, "lon": -73.9870, "coverage": 200, "orientation": 90},
        ]
    )
    route = [
        (40.7527, -73.9772),
        (40.7540, -73.9800),
        (40.7555, -73.9850),
        (40.7570, -73.9875),
        (40.7585, -73.9890),
        (40.7595, -73.9895),
        (40.7605, -73.9900),
    ]
else:
    mode = st.sidebar.selectbox("Mobility Mode", ["Driving", "Walking", "Stationary"])
    speed = st.sidebar.slider("Speed (km/h)", 0, 120, 30)
    st.sidebar.markdown("### Antennas CSV")
    ant_file = st.sidebar.file_uploader("Upload antenna CSV", type="csv")
    if ant_file:
        antenna_df = pd.read_csv(ant_file)
    else:
        antenna_df = pd.DataFrame(
            [{"lat": 40.7128, "lon": -74.0060, "coverage": 300, "orientation": 0}]
        )
    st.sidebar.markdown("### Route CSV")
    route_file = st.sidebar.file_uploader("Upload route CSV", type="csv")
    if route_file:
        df_rt = pd.read_csv(route_file)
        route = list(zip(df_rt.lat, df_rt.lon))
    else:
        route = [(antenna_df.lat.mean(), antenna_df.lon.mean())]

# Deck.gl 3D visualization
buildings = pdk.Layer(
    "FillExtrusionLayer",
    data="mapbox://mapbox.3d-buildings",
    get_fill_extrusion_height="properties.height",
    get_fill_extrusion_base="properties.min_height",
    get_fill_extrusion_color=[200, 200, 200],
    pickable=False,
)

antenna_layer = pdk.Layer(
    "ColumnLayer",
    data=antenna_df,
    get_position=["lon", "lat"],
    get_elevation="coverage",
    elevation_scale=1,
    radius=50,
    get_fill_color=[0, 128, 255, 100],
    pickable=True,
)

route_df = pd.DataFrame(route, columns=["lat", "lon"])
line_layer = pdk.Layer(
    "LineLayer",
    data=route_df,
    get_source_position=["lon", "lat"],
    get_target_position=["lon", "lat"],
    get_width=4,
    get_color=[255, 0, 0],
)

view_state = pdk.ViewState(
    latitude=route[0][0], longitude=route[0][1], zoom=14, pitch=60, bearing=0
)

pdk.settings.mapbox_api_key = MAPBOX_API_KEY
deck = pdk.Deck(
    layers=[buildings, antenna_layer, line_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v10",
)

st.pydeck_chart(deck, use_container_width=True)

# Prediction at current step
step = st.sidebar.slider("Step", 0, len(route) - 1, 0)
lat, lon = route[step]


def assemble(lat, lon, speed, mode, antenna_df):
    arr = [lat, lon, speed, encode[mode]]
    for _, r in antenna_df.iterrows():
        arr.extend([r.lat, r.lon, r.orientation])
    return np.array(arr)


seq = np.tile(assemble(lat, lon, speed, mode, antenna_df), (SEQ_LEN, 1))
nf = NF_FEATURES
if seq.shape[1] < nf:
    seq = np.hstack([seq, np.zeros((SEQ_LEN, nf - seq.shape[1]))])
else:
    seq = seq[:, :nf]

scaled = scaler.transform(seq).reshape(1, SEQ_LEN, nf)
raw_pred, smooth_pred = model.predict(scaled)
r, s = float(raw_pred), float(smooth_pred)
pred = r if abs(r - s) < abs(s - r) else s

st.metric("Predicted Throughput (Mbps)", f"{pred:.2f}")
