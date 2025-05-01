<<<<<<< HEAD
from dotenv import load_dotenv
load_dotenv()

import os
=======
import os
# Disable oneDNN optimizations to prevent name_scope stack errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

>>>>>>> 58004f7dd (Initial commit: MLOps pipeline, Streamlit apps, Docker setup)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
<<<<<<< HEAD
import pydeck as pdk
import tensorflow as tf
from tensorflow.keras import layers, models

# ─── Configuration ─────────────────────────────────────────────────────────────
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
=======
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin
import tensorflow as tf
from tensorflow.keras import layers, models

# Paths to artifacts
WEIGHTS_H5 = 'models/throughput_weights.weights.h5'
SCALER_PATH = 'models/scaler.gz'

# Rebuild CNN+LSTM model architecture without compile to avoid compile-time name_scope
def build_cnn_lstm_model(seq_len, nf):
>>>>>>> 58004f7dd (Initial commit: MLOps pipeline, Streamlit apps, Docker setup)
    inp = layers.Input(shape=(seq_len, nf))
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64)(x)
<<<<<<< HEAD
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
=======
    raw_out = layers.Dense(1, name='raw')(x)
    smooth_out = layers.Dense(1, name='smooth')(x)
    return models.Model(inputs=inp, outputs=[raw_out, smooth_out])

# Load scaler and determine expected features
scaler = joblib.load(SCALER_PATH)
SEQ_LEN = 10
NF_EXPECTED = int(scaler.scale_.shape[0])

# Build model and load weights
model = build_cnn_lstm_model(SEQ_LEN, NF_EXPECTED)
try:
    model.load_weights(WEIGHTS_H5)
except Exception:
    st.error('Model weights not found. Please run train_tf_model.py to generate throughput_weights.weights.h5')
    st.stop()

# Encode mobility modes
encode = {'Stationary': 0, 'Walking': 1, 'Driving': 2}

# Streamlit configuration
st.set_page_config(layout='wide', page_title='5G Throughput Demo Scenario')
st.title('🚀 5G Throughput Simulator Demo')

# Demo scenario selection
preset = st.sidebar.selectbox('Demo Scenario', ['-- Select --', 'Urban Drive'])
if preset == 'Urban Drive':
    scenario = 'Driving'
    speed = 60
    ant_df = pd.DataFrame([
        {'lat': 40.730610, 'lon': -73.935242, 'type': '5G SA mmWave', 'orientation': 45},
        {'lat': 40.731610, 'lon': -73.935742, 'type': '5G SA mmWave', 'orientation': 135},
        {'lat': 40.732610, 'lon': -73.936242, 'type': '5G NSA',        'orientation': 90}
    ])
    route = [
        (40.730610, -73.935242),
        (40.730800, -73.935200),
        (40.731000, -73.935150),
        (40.731200, -73.935100),
        (40.731400, -73.935050),
        (40.731600, -73.935000)
    ]
    beamwidth = 60
    coverage_m = 300
else:
    # Custom controls
    scenario = st.sidebar.selectbox('Mobility Mode', ['Stationary', 'Walking', 'Driving'])
    speed = st.sidebar.slider('Speed (km/h)', 0, 120, 30)
    st.sidebar.markdown('### Antenna Configuration')
    ant_file = st.sidebar.file_uploader('Upload antenna CSV', type=['csv'])
    if ant_file:
        ant_df = pd.read_csv(ant_file)
    else:
        ant_df = pd.DataFrame([
            {'lat': 40.7128, 'lon': -74.0060, 'type': '5G SA mmWave', 'orientation': 0},
            {'lat': 40.7138, 'lon': -74.0050, 'type': '5G NSA',        'orientation': 90}
        ])
    st.sidebar.markdown('### mmWave Coverage Settings')
    beamwidth = st.sidebar.slider('Beamwidth (°)', 10, 180, 90)
    coverage_m = st.sidebar.slider('Coverage Radius (m)', 100, 1000, 500)
    st.sidebar.markdown('### Route')
    route_file = st.sidebar.file_uploader('Upload Route CSV (lat,lon)', type=['csv'])
    if route_file:
        df_route = pd.read_csv(route_file)
        route = list(zip(df_route.lat, df_route.lon))
    else:
        route = [(ant_df.lat.mean(), ant_df.lon.mean())]

# Session state for navigation
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'route' not in st.session_state:
    st.session_state.route = route

# Step navigation buttons
col1, col2 = st.sidebar.columns(2)
if col1.button('Previous Step'):
    st.session_state.step = max(0, st.session_state.step - 1)
if col2.button('Next Step'):
    st.session_state.step = min(len(st.session_state.route) - 1, st.session_state.step + 1)

# Current position
lat, lon = st.session_state.route[st.session_state.step]
st.sidebar.markdown(f'**Step {st.session_state.step+1}/{len(st.session_state.route)}**')
st.sidebar.markdown(f'Position: ({lat:.6f}, {lon:.6f})')

# Create map
m = folium.Map(location=[lat, lon], zoom_start=16, tiles='CartoDB positron')
folium.PolyLine(st.session_state.route, color='blue', weight=2.5, opacity=0.7).add_to(m)
icon = 'car' if scenario == 'Driving' else 'male' if scenario == 'Walking' else 'circle'
folium.Marker([lat, lon], icon=folium.Icon(icon=icon, prefix='fa', color='red')).add_to(m)

# Plot antennas & coverage
for _, row in ant_df.iterrows():
    folium.Marker(
        [row.lat, row.lon],
        icon=folium.Icon(icon='signal', prefix='fa', color='blue'),
        popup=f"{row.type} @ {row.orientation}°"
    ).add_to(m)
    if 'mmwave' in row.type.lower():
        pts = []
        half = beamwidth / 2
        for ang in np.linspace(row.orientation - half, row.orientation + half, 30):
            r = radians(ang)
            d = coverage_m / 111000
            pts.append([row.lat + d * cos(r), row.lon + d * sin(r)])
        pts.insert(0, [row.lat, row.lon])
        folium.Polygon(pts, color='cyan', fill=True, fill_opacity=0.2).add_to(m)

st_folium(m, width=800, height=600)

# Feature assembly and prediction
def assemble_features(lat, lon, speed, ant_df, scenario):
    arr = [lat, lon, speed, encode[scenario]]
    for _, r in ant_df.iterrows():
        arr.extend([r.lat, r.lon, r.orientation])
    return np.array(arr)

seq = np.tile(assemble_features(lat, lon, speed, ant_df, scenario), (SEQ_LEN, 1))
# Pad or truncate to expected feature size
if seq.shape[1] < NF_EXPECTED:
    pad = np.zeros((SEQ_LEN, NF_EXPECTED - seq.shape[1]))
    seq = np.hstack([seq, pad])
elif seq.shape[1] > NF_EXPECTED:
    seq = seq[:, :NF_EXPECTED]

scaled = scaler.transform(seq).reshape(1, SEQ_LEN, -1)
pred_raw, pred_smooth = model.predict(scaled)
r, sv = float(pred_raw), float(pred_smooth)
pred = r if abs(r - sv) < abs(sv - r) else sv
st.metric('Predicted Throughput (Mbps)', f'{pred:.2f}')
>>>>>>> 58004f7dd (Initial commit: MLOps pipeline, Streamlit apps, Docker setup)
