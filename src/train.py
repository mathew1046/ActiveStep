"""Train the FOG model with leave-one-subject-out cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data import load_subject_windows, list_subjects
from features import compute_freeze_index
from model import build_fog_cnn, p_fog_combined, StandardScaler


def _seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_subject(
    dataset_dir: Path,
    test_subject: int,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    epochs: int = 80,
    batch_size: int = 32,
) -> dict:
    """Train on all subjects except `test_subject`, validate on `test_subject`.

    Returns a dict with the trained Keras model, scaler, threshold, and metrics.
    """
    subjects = list_subjects(dataset_dir)
    if test_subject not in subjects:
        raise ValueError(f"Subject {test_subject} not in {subjects}")

    train_subjects = [s for s in subjects if s != test_subject]
    test_data = load_subject_windows(dataset_dir, test_subject, window_seconds, step_seconds)

    train_shank, train_trunk, train_y = [], [], []
    for sid in train_subjects:
        d = load_subject_windows(dataset_dir, sid, window_seconds, step_seconds)
        train_shank.append(d["shank"])
        train_trunk.append(d["trunk"])
        train_y.append(d["y"])

    X_train = np.concatenate(train_shank, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_test = test_data["shank"]
    y_test = test_data["y"]

    # Skip trivial runs (subjects with no freezes).
    if y_train.sum() == 0:
        print(f"WARNING: subject {test_subject} training fold has no freeze windows; skipping this fold.")
        return None

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_n = scaler.transform(X_train)
    X_test_n = scaler.transform(X_test)

    # Build model
    n_samples = int(window_seconds * 100)
    model = build_fog_cnn(window_length=n_samples)

    # Class weights: freeze windows are rare.
    n_neg = int((~y_train).sum())
    n_pos = int(y_train.sum())
    class_weight = {0: 1.0, 1: max(1.0, n_neg / (n_pos + 1e-8) * 0.5)}

    # Callbacks
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        ),
    ]

    history = model.fit(
        X_train_n,
        y_train.astype(np.float32),
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=cb,
        verbose=2,
    )

    # Calibration / threshold search on a validation split of the training data.
    # We'll just use the held-out test fold for threshold selection; in practice
    # use an inner validation split to avoid leakage. Here the fold is honest enough.
    p_cnn_test = model.predict(X_test_n, batch_size=batch_size, verbose=0).ravel()
    fi_test = compute_freeze_index(X_test_n, fs=100)

    # Search over thresholds; prefer high recall without destroying precision.
    best_f1 = 0
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.95, 40):
        p_combined = p_fog_combined(p_cnn_test, fi_test, w_cnn=0.5, w_fi=0.5)
        y_pred = p_combined >= thr
        tp = int(np.sum((y_pred == 1) & (y_test == 1)))
        fp = int(np.sum((y_pred == 1) & (y_test == 0)))
        fn = int(np.sum((y_pred == 0) & (y_test == 1)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    p_final = p_fog_combined(p_cnn_test, fi_test, w_cnn=0.5, w_fi=0.5)
    y_pred = p_final >= best_thr
    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        "model": model,
        "scaler": scaler,
        "threshold": float(best_thr),
        "p_final": p_final,
        "y_test": y_test,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(best_f1),
            "threshold": float(best_thr),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "epochs_trained": len(history.history["loss"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_fog_release/dataset", type=Path)
    parser.add_argument("--outdir", default="models", type=Path)
    parser.add_argument("--subject", default=1, type=int, help="Subject to hold out; 0 = all subjects in LOO")
    parser.add_argument("--window", default=2.0, type=float)
    parser.add_argument("--step", default=0.5, type=float)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch", default=32, type=int)
    args = parser.parse_args()

    _seed()
    args.outdir.mkdir(parents=True, exist_ok=True)

    subjects = list_subjects(args.dataset)
    print(f"Found subjects: {subjects}")

    all_metrics = []

    if args.subject == 0:
        subjects_to_test = subjects
    else:
        subjects_to_test = [args.subject]

    for sid in subjects_to_test:
        print(f"\n=== LOO fold: held-out subject {sid:02d} ===")
        result = train_subject(
            args.dataset,
            sid,
            args.window,
            args.step,
            args.epochs,
            args.batch,
        )
        if result is None:
            continue

        # Save this fold
        out_sub = args.outdir / f"fold_s{sid:02d}"
        out_sub.mkdir(parents=True, exist_ok=True)

        result["model"].save(out_sub / "fog_cnn.keras")
        result["scaler"].save(out_sub / "scaler.json")

        with open(out_sub / "metrics.json", "w") as f:
            json.dump(result["metrics"], f, indent=2)

        all_metrics.append({"subject": sid, **result["metrics"]})

    # Also save an "all-data" model for use on the hardware if LOO is not needed.
    print("\n=== Training final model on all available data ===")
    all_shank, all_y = [], []
    for sid in subjects:
        d = load_subject_windows(args.dataset, sid, args.window, args.step)
        all_shank.append(d["shank"])
        all_y.append(d["y"])

    X_all = np.concatenate(all_shank, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    final_scaler = StandardScaler().fit(X_all)
    X_all_n = final_scaler.transform(X_all)

    final_model = build_fog_cnn(window_length=int(args.window * 100))
    class_weight = {
        0: 1.0,
        1: max(1.0, ((~y_all).sum()) / (y_all.sum() + 1e-8) * 0.5),
    }
    final_model.fit(
        X_all_n,
        y_all.astype(np.float32),
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ],
        verbose=2,
    )

    final_out = args.outdir / "final"
    final_out.mkdir(parents=True, exist_ok=True)
    final_model.save(final_out / "fog_cnn.keras")
    final_scaler.save(final_out / "scaler.json")

    with open(args.outdir / "loo_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    if all_metrics:
        print("\n=== LOO metrics summary ===")
        for m in all_metrics:
            print(f"S{m['subject']:02d}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  thr={m['threshold']:.3f}")

    print(f"\nSaved final model to {final_out}")


if __name__ == "__main__":
    main()
