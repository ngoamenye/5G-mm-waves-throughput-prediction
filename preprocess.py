import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

<<<<<<< HEAD
def load_data(path):
    """
    Load raw CSV data, handle missing values, and drop unused columns.
    """
    df = pd.read_csv(path)
    # Drop non-numeric or classification column if present
    if 'debit_class' in df.columns:
        df = df.drop(columns=['debit_class'])
    # Forward-fill then back-fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def build_sequences(df, seq_len):
    """
    Build rolling window sequences for model input and corresponding targets.
    Returns:
      X: np.array of shape (n_samples, seq_len, n_features)
      y_raw: np.array of raw throughput targets
      y_smooth: np.array of smooth throughput targets
      runs: np.array of sequence end indices (or run IDs)
      feature_names: list of column names used as features
    """
    # Define target and features
    target_raw = 'debit_brut'
    target_smooth = 'debit_lisse'
    feature_names = [c for c in df.columns if c not in [target_raw, target_smooth]]
    
    X, y_raw, y_smooth, runs = [], [], [], []
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len]
        X.append(seq[feature_names].values)
        end = df.iloc[i+seq_len]
        y_raw.append(end[target_raw])
        y_smooth.append(end[target_smooth])
        runs.append(i + seq_len)
    X = np.array(X)
    y_raw = np.array(y_raw)
    y_smooth = np.array(y_smooth)
    runs = np.array(runs)
    return X, y_raw, y_smooth, runs, feature_names

def scale_features(X):
    """
    Scale features to [0,1] across all samples and timesteps using MinMaxScaler.
    Returns scaled X and the fitted scaler.
    """
    n_samples, seq_len, n_features = X.shape
    scaler = MinMaxScaler()
    # Flatten for scaling
    X_flat = X.reshape(-1, n_features)
    X_scaled_flat = scaler.fit_transform(X_flat)
    # Restore shape
    X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_features)
=======
def load_data(path: str) -> pd.DataFrame:
    """Loads CSV and drops the 'debit_class' column if present."""
    df = pd.read_csv(path)
    if 'debit_class' in df.columns:
        df = df.drop(columns=['debit_class'])
    return df

def build_sequences(df: pd.DataFrame, seq_len: int):
    """Builds run-wise rolling windows of length seq_len."""
    features = df.select_dtypes(include=[np.number]).columns.drop(
        ['debit_brut', 'debit_lisse', 'run_num']
    ).tolist()
    X, y_raw, y_smooth, runs = [], [], [], []
    for run_id, grp in df.groupby('run_num'):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp) - seq_len):
            X.append(grp.loc[i:i+seq_len-1, features].values)
            y_raw.append(grp.loc[i+seq_len, 'debit_brut'])
            y_smooth.append(grp.loc[i+seq_len, 'debit_lisse'])
            runs.append(run_id)
    return np.stack(X), np.array(y_raw), np.array(y_smooth), np.array(runs), features

def scale_features(X: np.ndarray, scaler: MinMaxScaler = None):
    """Scales X with MinMaxScaler; fits a new scaler if none provided."""
    ns, seq_len, nf = X.shape
    X_flat = X.reshape(-1, nf)
    if scaler is None:
        scaler = MinMaxScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)
    X_scaled = X_flat.reshape(ns, seq_len, nf)
>>>>>>> 58004f7dd (Initial commit: MLOps pipeline, Streamlit apps, Docker setup)
    return X_scaled, scaler
