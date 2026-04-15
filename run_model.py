#!/usr/bin/env python3
"""
CLI inference script for the Vélib model.

Usage:
    python run_model.py --input "[30, 5, 10, 3, 2]"

Features order: capacity, numbikesavailable, numdocksavailable, mechanical, ebike
"""

import argparse
import json
from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path("artifacts/model.pkl")
FEATURE_NAMES = ["capacity", "numbikesavailable", "numdocksavailable", "mechanical", "ebike"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def main():
    parser = argparse.ArgumentParser(description="Predict Vélib station bike availability.")
    parser.add_argument(
        "--input",
        required=True,
        help=(
            f"Feature list as JSON string.\n"
            f"Order: {FEATURE_NAMES}\n"
            f'Example: --input "[30, 5, 10, 3, 2]"'
        ),
    )
    args = parser.parse_args()

    try:
        features = json.loads(args.input)
    except json.JSONDecodeError:
        raise ValueError('Invalid input. Use JSON list, e.g. --input "[30, 5, 10, 3, 2]"')

    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features ({FEATURE_NAMES}), got {len(features)}"
        )

    X = np.array(features).reshape(1, -1)
    model = load_model()

    pred = model.predict(X)
    proba = model.predict_proba(X)[0].tolist()

    result = {
        "prediction": int(pred[0]),
        "label": "bike_available" if pred[0] == 1 else "no_bike",
        "probability": {
            "no_bike": round(proba[0], 4),
            "bike_available": round(proba[1], 4),
        },
        "input_features": dict(zip(FEATURE_NAMES, features)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
