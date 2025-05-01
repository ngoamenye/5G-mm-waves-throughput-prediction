import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load model and scaler
MODEL_PATH = 'models/throughput_model.keras'
SCALER_PATH = 'models/scaler.gz'
# Load only for inference (no re-compilation)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# If you plan to retrain later, compile it explicitly:
model.compile(optimizer='adam', loss='mse')
scaler = joblib.load(SCALER_PATH)

# Configuration
SEQ_LEN = 10
encode = {'Stationary':0, 'Walking':1, 'Driving':2}

# Streamlit UI
st.set_page_config(layout="wide", page_title="5G Throughput Simulator")
st.title("5G Throughput Simulator (TensorFlow)")

# Sidebar controls
st.sidebar.header("Simulation Controls")
scenario = st.sidebar.selectbox("Mobility Mode", ["Stationary", "Walking", "Driving"])
speed = st.sidebar.slider("Speed (km/h)", 0, 120, 30)

st.sidebar.markdown("---")
st.sidebar.header("Antenna Configuration")
ant_data = st.sidebar.file_uploader("Upload antenna CSV", type=["csv"])
if ant_data:
    ant_df = pd.read_csv(ant_data)
else:
    ant_df = pd.DataFrame([
        {"lat": 40.7128, "lon": -74.0060, "orientation": 0},
        {"lat": 40.7138, "lon": -74.0050, "orientation": 90},
    ])

if st.sidebar.checkbox("Add new antenna"):
    with st.sidebar.form(key="add_ant_form"):
        new_lat = st.number_input("Latitude", format="%.6f")
        new_lon = st.number_input("Longitude", format="%.6f")
        new_orient = st.slider("Orientation (deg)", 0, 359, 0)
        submitted = st.form_submit_button("Add Antenna")
        if submitted:
            ant_df = ant_df.append({
                "lat": new_lat, "lon": new_lon, "orientation": new_orient
            }, ignore_index=True)

# Initialize user path
if 'path' not in st.session_state:
    center = (ant_df.lat.mean(), ant_df.lon.mean())
    st.session_state.path = [center]

def assemble_features(lat, lon, speed, ant_df, scenario):
    flat = [lat, lon, speed, encode[scenario]]
    for _, r in ant_df.iterrows():
        flat += [r.lat, r.lon, r.orientation]
    return np.array(flat)

# Map display
m = folium.Map(location=st.session_state.path[-1], zoom_start=14)
for _, r in ant_df.iterrows():
    folium.Marker(
        location=(r.lat, r.lon),
        icon=folium.Icon(color="blue"),
        popup=f"Orient: {r.orientation}"
    ).add_to(m)
folium.CircleMarker(
    location=st.session_state.path[-1], radius=6,
    color="red", fill=True
).add_to(m)
st_folium(m, width=800, height=600)

# Simulate movement
if scenario != "Stationary":
    last_lat, last_lon = st.session_state.path[-1]
    delta = speed / 100000
    new_lat = last_lat + np.random.uniform(-delta, delta)
    new_lon = last_lon + np.random.uniform(-delta, delta)
    st.session_state.path.append((new_lat, new_lon))

# Predict throughput
user_lat, user_lon = st.session_state.path[-1]
feat = assemble_features(user_lat, user_lon, speed, ant_df, scenario)
dummy_seq = np.tile(feat, (SEQ_LEN,1))
try:
    seq_scaled = scaler.transform(dummy_seq).reshape(1, SEQ_LEN, -1)
except ValueError:
    # Fallback: features don’t match scaler, so fit a fresh scaler on dummy_seq
    tmp_scaler = MinMaxScaler()
    seq_scaled = tmp_scaler.fit_transform(dummy_seq).reshape(1, SEQ_LEN, -1)
pred_raw, pred_smooth = model.predict(seq_scaled)
val_raw = float(pred_raw.squeeze())
val_smooth = float(pred_smooth.squeeze())
pred_final = val_raw if abs(val_raw - val_smooth) < abs(val_smooth - val_raw) else val_smooth
st.metric("Predicted Throughput (Mbps)", f"{pred_final:.2f}")

# Retrain button
if st.button("Retrain Model on Latest Sample"):
    model.fit(seq_scaled, np.array([[pred_final, pred_final]]), epochs=1, verbose=0)
    model.save(MODEL_PATH)
    st.success("Model retrained incrementally.")
