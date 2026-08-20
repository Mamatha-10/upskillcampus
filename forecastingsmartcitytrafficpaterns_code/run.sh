#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Use python3 from PATH; adjust if you prefer a specific interpreter
python3 train_models.py
python3 scripts/generate_report.py

echo "Done. Generated: models/traffic_prediction_model.pkl, data/submission.csv, reports/"
