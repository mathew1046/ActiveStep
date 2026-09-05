"""Evaluate a trained model/fold: latency, precision, recall, F1, and event-level metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import load_subject_windows, list_subjects
from features import compute_freeze_index
from model import StandardScaler, p_fog_combined


def event_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, min_gap: int = 5) -> dict:
    """Count how many freeze events (contiguous 2s windows) were detected.

    An event is a contiguous block of freeze labels. It is "detected" if at
    least one window inside it is predicted positive. This is much more
    clinically relevant than per-window accuracy.
    """
    # Find freeze events in ground truth.
    events = []
    in_event = False
    start = 0
    for i, v in enumerate(y_true):
        if v and not in_event:
            in_event = True
            start = i
        elif not v and in_event:
            events.append((start, i))
            in_event = False
    if in_event:
        events.append((start, len(y_true)))

    detected = 0
    for s, e in events:
        if np.any(y_pred[s:e]):
            detected += 1

    # False freezes: contiguous blocks of predictions outside true events.
    fp_events = 0
    in_fp = False
    for i, p in enumerate(y_pred):
        # inside a true event? then predictions are expected.
        in_true_event = any(s <= i < e for s, e in events)
        if p and not in_true_event and not in_fp:
            in_fp = True
        elif (not p or in_true_event) and in_fp:
            in_fp = False
            fp_events += 1
    if in_fp:
        fp_events += 1

    event_recall = detected / (len(events) + 1e-8)
    event_fp_rate = fp_events / (len(events) + 1e-8)
    return {
        "n_true_events": int(len(events)),
        "n_detected_events": int(detected),
        "event_recall": float(event_recall),
        "n_false_event_blocks": int(fp_events),
        "false_event_rate": float(event_fp_rate),
    }


def detection_latency(p_scores: np.ndarray, y_true: np.ndarray, step_seconds: float = 0.5) -> float:
    """Mean seconds from the first true window to the first positive prediction inside an event."""
    events = []
    in_event = False
    start = 0
    for i, v in enumerate(y_true):
        if v and not in_event:
            in_event = True
            start = i
        elif not v and in_event:
            events.append((start, i))
            in_event = False
    if in_event:
        events.append((start, len(y_true)))

    latencies = []
    for s, e in events:
        idx = np.where(p_scores[s:e] >= 0.5)[0]
        if len(idx):
            latencies.append((idx[0] + s) * step_seconds - s * step_seconds)
    return float(np.mean(latencies)) if latencies else 0.0


def evaluate_fold(
    dataset_dir: Path,
    model_dir: Path,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> dict:
    """Evaluate every held-out fold in `model_dir/fold_s**` plus the final model."""
    subjects = list_subjects(dataset_dir)
    all_metrics = {}

    for sid in subjects:
        fold_dir = model_dir / f"fold_s{sid:02d}"
        if not fold_dir.exists():
            continue

        model = tf.keras.models.load_model(fold_dir / "fog_cnn.keras")
        scaler = StandardScaler().load(fold_dir / "scaler.json")
        with open(fold_dir / "metrics.json") as f:
            fold_meta = json.load(f)
        threshold = fold_meta["threshold"]

        test = load_subject_windows(dataset_dir, sid, window_seconds, step_seconds)
        X_test = scaler.transform(test["shank"])
        y_test = test["y"]

        p_cnn = model.predict(X_test, batch_size=32, verbose=0).ravel()
        fi = compute_freeze_index(X_test, fs=100)
        p_final = p_fog_combined(p_cnn, fi, w_cnn=0.5, w_fi=0.5)
        y_pred = p_final >= threshold

        tp = int(np.sum((y_pred == 1) & (y_test == 1)))
        fp = int(np.sum((y_pred == 1) & (y_test == 0)))
        fn = int(np.sum((y_pred == 0) & (y_test == 1)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        em = event_level_metrics(y_test, y_pred)
        lat = detection_latency(p_final, y_test, step_seconds)

        all_metrics[f"S{sid:02d}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
            "event_recall": em["event_recall"],
            "false_event_rate": em["false_event_rate"],
            "mean_detection_latency_s": float(lat),
        }

    # Summary
    pr = [m["precision"] for m in all_metrics.values()]
    rc = [m["recall"] for m in all_metrics.values()]
    f1s = [m["f1"] for m in all_metrics.values()]
    all_metrics["summary"] = {
        "mean_precision": float(np.mean(pr)),
        "mean_recall": float(np.mean(rc)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
    }
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_fog_release/dataset", type=Path)
    parser.add_argument("--model-dir", default="models", type=Path)
    parser.add_argument("--out", default="models/eval_metrics.json", type=Path)
    parser.add_argument("--window", default=2.0, type=float)
    parser.add_argument("--step", default=0.5, type=float)
    args = parser.parse_args()

    metrics = evaluate_fold(args.dataset, args.model_dir, args.window, args.step)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
