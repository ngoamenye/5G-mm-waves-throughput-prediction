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

# ─── Configuration ────────────────────────────────────────────────
DATA_PATH    = 'data/mm-5G-enriched.csv'
MODEL_DIR    = 'models'
SCALER_PATH  = os.path.join(MODEL_DIR, 'scaler.gz')
WEIGHTS_H5   = os.path.join(MODEL_DIR, 'throughput_weights.weights.h5')
FEATURES_PKL = os.path.join(MODEL_DIR, 'feature_names.pkl')
SEQ_LEN      = 10

# Load data and pick 3 example runs
df = pd.read_csv(DATA_PATH)
runs = sorted(df['run_num'].unique())[:3]

# Load scaler, model, and feature names
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PKL)
nf = len(feature_names)

def build_model(seq_len, nf):
    inp = layers.Input(shape=(seq_len, nf))
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64)(x)
    raw = layers.Dense(1, name='raw')(x)
    smooth = layers.Dense(1, name='smooth')(x)
    return models.Model(inputs=inp, outputs=[raw, smooth])

model = build_model(SEQ_LEN, nf)
model.load_weights(WEIGHTS_H5)

# Sidebar: choose run
st.sidebar.title("Simulation Controls")
run_sel = st.sidebar.selectbox("Select Run", runs)
df_run = df[df['run_num'] == run_sel].reset_index(drop=True)

# Sidebar: step slider
step = st.sidebar.slider("Step Index", 0, len(df_run)-1, 0)

# Extract step features window
start = max(0, step - SEQ_LEN + 1)
seq_df = df_run.loc[start:step, feature_names]
# pad at top if needed
if len(seq_df) < SEQ_LEN:
    pad = pd.DataFrame([seq_df.iloc[0]] * (SEQ_LEN - len(seq_df)), columns=feature_names)
    seq_df = pd.concat([pad, seq_df], ignore_index=True)

# Scale and reshape
X_seq = seq_df.values
X_flat = X_seq.reshape(-1, nf)
X_scaled = scaler.transform(X_flat).reshape(1, SEQ_LEN, nf)

# Predict
pred_raw, pred_smooth = model.predict(X_scaled)
raw_val, smooth_val = pred_raw.squeeze(), pred_smooth.squeeze()
pred = raw_val if abs(raw_val - smooth_val) < abs(smooth_val - raw_val) else smooth_val

# Detect unknown features (outside scaler range)
mins = scaler.data_min_
maxs = scaler.data_max_
unknown = ((X_seq < mins) | (X_seq > maxs)).any()
if unknown:
    st.error("⚠️ Unknown feature detected: queuing sample for retraining.")
    # Here you could append df_run.loc[step] to a retraining buffer

# UI: display metrics
st.title(f"🚦 Run {run_sel} – Step {step}")
lat = df_run.loc[step, 'latitude']
lon = df_run.loc[step, 'longitude']
speed = df_run.loc[step, 'movingSpeed']

st.subheader("Prediction")
st.write(f"Raw: {raw_val:.2f} Mbps | Smooth: {smooth_val:.2f} Mbps")
st.metric("Final Throughput (Mbps)", f"{pred:.2f}")

# Define antenna positions & coverage radii (m)
antennas = df[['tower_id','latitude','longitude']].drop_duplicates().set_index('tower_id').reset_index()
# Example coverage radii
antennas['mm_radius'] = 200
antennas['lte_radius'] = 500

# Build map layers
pdk.settings.mapbox_api_key = os.getenv("MAPBOX_API_KEY")

# Heatmap of throughput around the step
grid = []
for dlat in np.linspace(lat-0.005, lat+0.005, 15):
    for dlon in np.linspace(lon-0.005, lon+0.005, 15):
        base = []
        for fn in feature_names:
            if fn in ['latitude','longitude','movingSpeed','compassDirection']:
                # simple spatial interpolation
                if fn == 'latitude': base.append(dlat)
                elif fn == 'longitude': base.append(dlon)
                elif fn == 'movingSpeed': base.append(speed)
                else: base.append(0)
            else:
                base.append(0)
        arr = np.array(base)
        if len(arr) < nf:
            arr = np.concatenate([arr, np.zeros(nf - len(arr))])
        Xg = np.tile(arr, (SEQ_LEN,1))
        Xg_scaled = scaler.transform(Xg.reshape(-1,nf)).reshape(1,SEQ_LEN,nf)
        r,g = model.predict(Xg_scaled)
        val = float(r) if abs(r-g)<abs(g-r) else float(g)
        grid.append({'lat':dlat,'lon':dlon,'throughput':val})
grid_df = pd.DataFrame(grid)

coverage = pdk.Layer(
    "HeatmapLayer",
    data=grid_df,
    get_position=["lon","lat"],
    get_weight="throughput",
    radiusPixels=50
)
mmwave_layer = pdk.Layer(
    "ColumnLayer",
    data=antennas,
    get_position=["longitude","latitude"],
    get_elevation="mm_radius",
    elevation_scale=1,
    radius="mm_radius",
    get_fill_color=[0,128,255,80],
    pickable=False
)
lte_layer = pdk.Layer(
    "ColumnLayer",
    data=antennas,
    get_position=["longitude","latitude"],
    get_elevation="lte_radius",
    elevation_scale=1,
    radius="lte_radius",
    get_fill_color=[255,165,0,50],
    pickable=False
)
point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame([{'lat':lat,'lon':lon}]),
    get_position=["lon","lat"],
    get_radius=10,
    get_fill_color=[255,0,0],
)

view = pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=45)
deck = pdk.Deck(
    layers=[coverage, mmwave_layer, lte_layer, point_layer],
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10"
)
st.pydeck_chart(deck, use_container_width=True)

# Recommendations
if pred < 20:
    st.warning("🔴 Very low throughput: recommend switching to LTE fallback.")
elif pred < 50:
    st.info("🟠 Moderate throughput: consider reorienting to closest antenna.")
else:
    st.success("🟢 Good coverage: continue on 5G mmWave.")
