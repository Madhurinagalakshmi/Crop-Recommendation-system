# src/models/baseline.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def run_baselines(csv_path: str, save_dir: str = "artifacts/"):
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values
    y = df['label_enc'].values

    # Proper train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Classes in train: {sorted(np.unique(y_train))}")
    print(f"Classes in test:  {sorted(np.unique(y_test))}\n")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf_preds))
    joblib.dump(rf, f"{save_dir}/random_forest.pkl")

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    print(classification_report(y_test, xgb_preds))
    joblib.dump(xgb, f"{save_dir}/xgboost.pkl")

    return {"random_forest": rf_acc, "xgboost": xgb_acc}

if __name__ == "__main__":
    results = run_baselines("data/processed/crop_processed.csv")
    print("\n=== BASELINE RESULTS ===")
    print(f"Random Forest: {results['random_forest']:.4f}")
    print(f"XGBoost:       {results['xgboost']:.4f}")