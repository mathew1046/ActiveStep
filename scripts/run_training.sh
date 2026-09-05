#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source /opt/anaconda/etc/profile.d/conda.sh
conda activate activestep

echo "=== Training leave-one-subject-out models ==="
python src/train.py --dataset dataset_fog_release/dataset --subject 0 --outdir models --epochs 80 --batch 32

echo "=== Evaluating all folds ==="
python src/evaluate.py --dataset dataset_fog_release/dataset --model-dir models --out models/eval_metrics.json

echo "=== Quantizing final model ==="
python src/quantize.py --model models/final/fog_cnn.keras --scaler models/final/scaler.json --dataset dataset_fog_release/dataset --out models/final/model_quantized.tflite

echo "=== Done ==="
