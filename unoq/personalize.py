"""Personalization: baseline calibration, adaptive threshold, optional fine-tune + OTA."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .db import connect

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "final"


def calibrate(imu_windows: np.ndarray, fs: int = 100) -> dict:
    """30 s baseline calibration -> per-patient cadence, FI, amplitude baselines.

    `imu_windows`: (N, W, 3) array of normal-walking windows.
    """
    from src.features import compute_cadence, compute_freeze_index, step_amplitude

    cadences = [compute_cadence(w[:, 1], fs) for w in imu_windows]
    fis = compute_freeze_index(imu_windows, fs)
    amps = [step_amplitude(w[:, 1]) for w in imu_windows]
    baseline = {
        "cadence": float(np.median(cadences)),
        "fi": float(np.median(fis)),
        "step_amp": float(np.median(amps)),
        "fi_mean": float(np.mean(fis)),
        "fi_std": float(np.std(fis)),
    }
    (MODEL_DIR / "baseline.json").write_text(json.dumps(baseline, indent=2))
    return baseline


def fit_threshold(labels: list[tuple[float, int]]) -> float:
    """Logistic fit of the decision threshold from (pfog, label) pairs.

    Works from ~10 labels. Returns the threshold that maximizes balanced
    accuracy (mean of sensitivity and specificity).
    """
    if len(labels) < 5:
        return 0.5  # not enough data — keep default
    scores = np.array([s for s, _ in labels])
    y = np.array([l for _, l in labels])
    best, best_bal = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 60):
        tp = ((scores >= thr) & (y == 1)).sum()
        tn = ((scores < thr) & (y == 0)).sum()
        fp = ((scores >= thr) & (y == 0)).sum()
        fn = ((scores < thr) & (y == 1)).sum()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        bal = (sens + spec) / 2
        if bal > best_bal:
            best, best_bal = float(thr), bal
    return best


def labels_from_db() -> list[tuple[float, int]]:
    conn = connect()
    rows = conn.execute(
        """SELECT e.pfog_peak, l.label FROM events e
           JOIN labels l ON l.event_id = e.id"""
    ).fetchall()
    return [(r["pfog_peak"], 1 if r["label"] == "TRUE" else 0) for r in rows]


def update_threshold_and_push(send_threshold_fn):
    """Re-fit the threshold from accumulated labels and push it to the ESP32."""
    thr = fit_threshold(labels_from_db())
    send_threshold_fn(thr)
    return thr


def finetune_last_layer(new_windows: np.ndarray, new_labels: np.ndarray, out_dir: Path = MODEL_DIR):
    """Optional: fine-tune only the final Dense layer, export TFLite for OTA."""
    import tensorflow as tf

    base = tf.keras.models.load_model(MODEL_DIR / "fog_cnn.keras")
    # Freeze everything except the last Dense layer.
    for layer in base.layers[:-1]:
        layer.trainable = False
    base.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy")
    base.fit(new_windows, new_labels, epochs=10, batch_size=16, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(base)
    tflite = converter.convert()
    out = out_dir / "model_v2.tflite"
    out.write_bytes(tflite)
    return out
