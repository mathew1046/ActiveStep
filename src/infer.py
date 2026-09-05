"""Run a single window through the quantized TFLite model (useful for ESP32 parity testing)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import load_subject_windows


def load_tflite(model_path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def tflite_predict(interpreter: tf.lite.Interpreter, window: np.ndarray) -> float:
    """Return pFOG from the quantized model. Handles both float and int8 IO."""
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]

    x = np.expand_dims(window, axis=0).astype(in_d["dtype"])
    if in_d["dtype"] == np.int8:
        scale, zp = in_d["quantization_parameters"]["scales"][0], in_d["quantization_parameters"]["zero_points"][0]
        x = np.clip(np.round(window / scale + zp), -128, 127).astype(np.int8)[None]

    interpreter.set_tensor(in_d["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(out_d["index"])[0, 0]
    if out_d["dtype"] == np.int8:
        scale, zp = out_d["quantization_parameters"]["scales"][0], out_d["quantization_parameters"]["zero_points"][0]
        return float((out.astype(np.float32) - zp) * scale)
    return float(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/final/model_quantized.tflite", type=Path)
    parser.add_argument("--meta", default="models/final/model_meta.json", type=Path)
    parser.add_argument("--dataset", default="dataset_fog_release/dataset", type=Path)
    parser.add_argument("--subject", default=1, type=int)
    parser.add_argument("--index", default=0, type=int, help="Window index to test")
    args = parser.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    data = load_subject_windows(args.dataset, args.subject)
    X = data["shank"][args.index]
    y = data["y"][args.index]

    X_scaled = (X - np.array(meta["scaler"]["mean"])) / np.array(meta["scaler"]["std"])

    interpreter = load_tflite(args.model)
    p = tflite_predict(interpreter, X_scaled)

    print(f"Subject {args.subject:02d}, window {args.index}, label={y}, pFOG(tflite)={p:.4f}")

    # Find the first freeze window and one non-freeze window for a sanity check.
    pos_idx = np.where(data["y"])[0]
    neg_idx = np.where(~data["y"])[0]
    if len(pos_idx):
        Xp = (data["shank"][pos_idx[0]] - np.array(meta["scaler"]["mean"])) / np.array(meta["scaler"]["std"])
        print(f"First freeze window: pFOG={tflite_predict(interpreter, Xp):.4f}")
    if len(neg_idx):
        Xn = (data["shank"][neg_idx[0]] - np.array(meta["scaler"]["mean"])) / np.array(meta["scaler"]["std"])
        print(f"First non-freeze window: pFOG={tflite_predict(interpreter, Xn):.4f}")


if __name__ == "__main__":
    main()
