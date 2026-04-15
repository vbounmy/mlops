"""
Training script for Vélib bike availability prediction.
Dataset: Vélib - Vélos et bornes - Disponibilité temps réel (opendata.paris.fr)

Goal: Predict whether a station has bikes available (binary classification)
Features: capacity, mechanical bikes, electrical bikes, available docks
Target: is_available (1 = at least 1 bike available, 0 = no bikes)
"""

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import json

DATASET_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "velib-disponibilite-en-temps-reel/exports/json?limit=10000"
)

def fetch_data() -> pd.DataFrame:
    """Fetch real-time Vélib data from Paris Open Data."""
    print("Fetching Vélib data from opendata.paris.fr ...")
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    records = resp.json()
    df = pd.DataFrame(records)
    print(f"  → {len(df)} stations fetched.")
    return df


def preprocess(df: pd.DataFrame):
    """Select and clean features, build target variable."""
    feature_cols = [
        "capacity",
        "numbikesavailable",
        "numdocksavailable",
        "mechanical",
        "ebike",
    ]

    # Keep only rows where all feature cols exist
    df = df.dropna(subset=feature_cols).copy()

    # Cast to numeric (API sometimes returns strings)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values

    # Target: 1 if at least one bike available, else 0
    y = (df["numbikesavailable"] > 0).astype(int).values

    return X, y, feature_cols


def main():
    df = fetch_data()
    X, y, feature_names = preprocess(df)

    print(f"Dataset shape: {X.shape}  |  Class balance: {y.mean():.2%} available")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["no_bike", "bike_available"]))

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(pipeline, model_path)

    metrics = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_names": feature_names,
    }
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved → {model_path}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    main()
