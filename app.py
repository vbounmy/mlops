"""
Flask API for Vélib bike availability prediction.

Endpoints:
  GET  /health   → service health check
  POST /predict  → predict if a station has bikes available

Example POST /predict body:
{
  "features": [30, 5, 10, 3, 2]
}
Features order: capacity, numbikesavailable, numdocksavailable, mechanical, ebike
"""

from flask import Flask, request, jsonify
import joblib
import os
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("artifacts/model.pkl")

FEATURE_NAMES = ["capacity", "numbikesavailable", "numdocksavailable", "mechanical", "ebike"]

if not MODEL_PATH.exists():
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": str(MODEL_PATH),
        "features": FEATURE_NAMES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({
            "error": "Send JSON with key 'features'",
            "expected_format": {
                "features": [capacity, numbikesavailable, numdocksavailable, mechanical, ebike]
            }
        }), 400

    features = data["features"]

    if len(features) != len(FEATURE_NAMES):
        return jsonify({
            "error": f"Expected {len(FEATURE_NAMES)} features, got {len(features)}",
            "feature_names": FEATURE_NAMES,
        }), 400

    try:
        pred = model.predict([features])
        proba = model.predict_proba([features])[0].tolist()
        return jsonify({
            "prediction": int(pred[0]),
            "label": "bike_available" if pred[0] == 1 else "no_bike",
            "probability": {
                "no_bike": round(proba[0], 4),
                "bike_available": round(proba[1], 4),
            },
            "input_features": dict(zip(FEATURE_NAMES, features)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
