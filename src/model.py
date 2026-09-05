"""Tiny 1D-CNN for FOG onset detection, plus a combiner for Freeze Index + learned pFOG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from src.features import p_fog_combined  # noqa: F401  (re-exported for compatibility)
except ImportError:  # imported as a top-level module (e.g. `python src/train.py`)
    from features import p_fog_combined  # noqa: F401


def build_fog_cnn(
    window_length: int,
    n_axes: int = 3,
    n_filters: tuple[int, int] = (16, 32),
    dense_units: int = 16,
    dropout: float = 0.2,
) -> keras.Model:
    """A small 1D-CNN designed to be quantized and run on the ESP32 (TFLite Micro).

    Input:  (window_length, n_axes)  float32
    Output: (1,) sigmoid freeze probability
    """
    inputs = keras.Input(shape=(window_length, n_axes), name="shank_acc")

    x = layers.Conv1D(n_filters[0], 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(n_filters[1], 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_fog")(x)

    model = keras.Model(inputs, outputs, name="fog_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class StandardScaler:
    """Per-axis scaler with optional trailing-window mean removal."""

    def __init__(self, demean_window: bool = False):
        self.mean: np.ndarray = None
        self.std: np.ndarray = None
        self.demean_window = bool(demean_window)

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return X - X.mean(axis=-2, keepdims=True) if self.demean_window else X

    def fit(self, X: np.ndarray):
        prepared = self._prepare(X)
        self.mean = prepared.reshape(-1, prepared.shape[-1]).mean(axis=0).astype(np.float32)
        self.std = prepared.reshape(-1, prepared.shape[-1]).std(axis=0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self._prepare(X) - self.mean) / (self.std + 1e-8)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * (self.std + 1e-8) + self.mean

    def save(self, path: str | Path):
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "demean_window": self.demean_window,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path):
        with open(path) as f:
            data = json.load(f)
        obj = cls(demean_window=data.get("demean_window", False))
        obj.mean = np.array(data["mean"], dtype=np.float32)
        obj.std = np.array(data["std"], dtype=np.float32)
        return obj
