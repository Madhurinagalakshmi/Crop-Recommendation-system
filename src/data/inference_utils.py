# src/data/inference_utils.py

import numpy as np
import torch
import joblib

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def preprocess_single_input(raw_input: dict, artifacts_dir: str = "artifacts/") -> torch.Tensor:
    """
    Called by Member 3's FastAPI endpoint before passing to GNN.
    
    raw_input example:
    {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.8, "humidity": 82.0,
        "ph": 6.5, "rainfall": 202.9
    }
    
    Returns: torch.Tensor of shape (1, 7)
    """
    scaler = joblib.load(f"{artifacts_dir}/scaler.pkl")
    values = [[raw_input[col] for col in FEATURE_COLS]]
    scaled = scaler.transform(values)
    return torch.tensor(scaled, dtype=torch.float)

def decode_prediction(label_idx: int, artifacts_dir: str = "artifacts/") -> str:
    """
    Converts integer prediction back to crop name.
    Called by Member 3 to return human-readable result.
    """
    le = joblib.load(f"{artifacts_dir}/label_encoder.pkl")
    return le.inverse_transform([label_idx])[0]