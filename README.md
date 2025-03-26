# 5G-mm-waves-throughput-prediction
"Comprehensive modeling and forecasting of 5G mmWave throughput using classical machine learning, advanced gradient boosting, deep learning (LSTM, CNN, TFT), and AutoML techniques. Includes smoothing and snapshot-based prediction strategies for enhanced accuracy."

## 🎯 Objectives
- Smooth throughput signal
- Predict with and without rolling window
- Compare models: MLP, CNN-LSTM, DeepAR, TFT, H2O AutoML
- Deploy as an API or dashboard

## 📁 Project Structure
- `notebooks/`: all modeling notebooks
- `data/`: processed datasets
- `models/`: saved trained models
- `images/`: visualizations
- `app/`: dashboard or API (optional)

## ⚙️ Techniques
- TensorFlow / Keras, PyTorch Forecasting
- Rolling window, EMA, SMA, Median smoothing
- AutoML (H2O)
- Temporal Fusion Transformer (TFT)

## 📊 Results
See [notebooks/tft_snapshot_model.ipynb](notebooks/tft_snapshot_model.ipynb) for best performing model.

## ▶️ How to run
```bash
pip install -r requirements.txt
