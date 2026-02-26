# src/data/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COL = 'label'

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load dataset, drop duplicates and nulls."""
    df = pd.read_csv(csv_path)
    print(f"Loaded: {df.shape}")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"After cleaning: {df.shape}")
    return df

def encode_labels(df: pd.DataFrame, save_dir: str = "artifacts/"):
    """Encode crop names to integers. Saves encoder for inference."""
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df[TARGET_COL])
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))
    print(f"Classes: {list(le.classes_)}")
    return df, le

def scale_features(df: pd.DataFrame, save_dir: str = "artifacts/"):
    """StandardScaler on features. Saves scaler for inference."""
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    return df, scaler

def split_data(df: pd.DataFrame):
    """Train/val/test split with stratification."""
    X = df[FEATURE_COLS].values
    y = df['label_enc'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_input(raw_input: dict, artifacts_dir: str = "artifacts/") -> np.ndarray:
    """
    For Member 3 (API): preprocess a single user input at inference time.
    raw_input = {'N': 90, 'P': 42, 'K': 43, 'temperature': 20.8,
                 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9}
    Returns scaled numpy array of shape (1, 7)
    """
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
    values = [[raw_input[col] for col in FEATURE_COLS]]
    return scaler.transform(values)

if __name__ == "__main__":
    df = load_and_clean("data/raw/crop_recommendation.csv")
    df, le = encode_labels(df)
    df, scaler = scale_features(df)
    df.to_csv("data/processed/crop_processed.csv", index=False)
    print("Preprocessing complete.")