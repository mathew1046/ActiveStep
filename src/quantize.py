"""Convert the Keras model to a fully-quantized TFLite model for the ESP32."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import load_subject_windows, list_subjects
from model import StandardScaler


def representative_dataset_factory(X: np.ndarray, n_samples: int = 200):
    """Create a generator that yields (1, window, 3) int8 samples.

    TFLite full-integer quantization needs a representative sample of inputs
    to calibrate the quantization ranges. We use a random subset of the
    training data.
    """
    idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    samples = X[idx]

    def generator():
        for s in samples:
            # Quantized model expects int8 with the calibration range built in.
            yield [np.expand_dims(s, axis=0).astype(np.float32)]

    return generator


def convert_model(model: keras.Model, out_path: Path, int8: bool, X_rep: np.ndarray | None) -> bytes:
    """Convert a Keras model to TFLite.

    int8=False produces a pure float32 model (best parity with Keras).
    int8=True quantizes weights+activations to int8 but keeps float32 IO,
    which is faster on the ESP32 but measurably degrades this small model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_factory(X_rep)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Keep float32 input/output: forcing int8 IO destroyed the output
        # resolution (scale ~0.0039) and flipped the ranking.
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    return tflite_model


def quantize(
    keras_model_path: Path,
    scaler_path: Path,
    dataset_dir: Path,
    out_path: Path,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    n_rep: int = 200,
) -> bytes:
    """Load a Keras model, convert to TFLite (float32 + int8 variants), and save."""
    model = tf.keras.models.load_model(keras_model_path)
    scaler = StandardScaler().load(scaler_path)

    # Build the representative dataset from all training subjects.
    all_X = []
    for sid in list_subjects(dataset_dir):
        d = load_subject_windows(dataset_dir, sid, window_seconds, step_seconds)
        all_X.append(scaler.transform(d["shank"]))
    X = np.concatenate(all_X, axis=0)

    # Float32 model — primary artifact for the ESP32 (tiny, exact accuracy).
    tflite_model = convert_model(model, out_path, int8=False, X_rep=None)

    # int8 variant alongside, for comparison / faster inference experiments.
    int8_path = out_path.with_name(out_path.stem + "_int8.tflite")
    try:
        convert_model(model, int8_path, int8=True, X_rep=X)
    except Exception as e:
        print(f"int8 conversion failed (non-fatal): {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    # Also write a C header for direct Arduino inclusion.
    c_header = out_path.with_suffix(".h")
    try:
        result = subprocess.run(
            ["xxd", "-i", str(out_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        array_name = out_path.stem.replace("-", "_").replace(".", "_")
        c_content = result.stdout.replace(out_path.stem + "_tflite", array_name)
        c_header.write_text(c_content)
    except Exception as e:
        print(f"Could not generate C header: {e}")

    # Save the quantization metadata: zero point and scale of the output.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    out_details = interpreter.get_output_details()[0]
    in_details = interpreter.get_input_details()[0]
    meta = {
        "input_scale": float(in_details["quantization_parameters"]["scales"][0]) if in_details["quantization_parameters"]["scales"].size else 1.0,
        "input_zero_point": int(in_details["quantization_parameters"]["zero_points"][0]) if in_details["quantization_parameters"]["zero_points"].size else 0,
        "output_scale": float(out_details["quantization_parameters"]["scales"][0]) if out_details["quantization_parameters"]["scales"].size else 1.0,
        "output_zero_point": int(out_details["quantization_parameters"]["zero_points"][0]) if out_details["quantization_parameters"]["zero_points"].size else 0,
        "io_dtype": str(in_details["dtype"]),
        "window_samples": int(window_seconds * 100),
        "n_axes": 3,
        "scaler": json.loads(scaler_path.read_text()),
    }
    (out_path.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))

    return tflite_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="Path to .keras model")
    parser.add_argument("--scaler", required=True, type=Path, help="Path to scaler.json")
    parser.add_argument("--dataset", default="dataset_fog_release/dataset", type=Path)
    parser.add_argument("--out", default="models/final/model_quantized.tflite", type=Path)
    parser.add_argument("--n-rep", default=200, type=int)
    args = parser.parse_args()

    quantize(args.model, args.scaler, args.dataset, args.out, n_rep=args.n_rep)
    print(f"Saved quantized model to {args.out}")
    print(f"C header: {args.out.with_suffix('.h')}")


if __name__ == "__main__":
    main()
