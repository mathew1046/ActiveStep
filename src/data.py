"""Daphnet FOG dataset loading, resampling, and sliding-window extraction."""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

# Daphnet is sampled at 64 Hz; we re-sample to 100 Hz to match the ESP32 rate.
ORIGINAL_FS = 64
TARGET_FS = 100

# Columns in each .txt file:
# 0: time (ms)
# 1-3: shank (ankle)  forward, vertical, lateral  [mg]
# 4-6: thigh         forward, vertical, lateral  [mg]
# 7-9: trunk         forward, vertical, lateral  [mg]
# 10:  annotation
SHANK_COLS = slice(1, 4)
THIGH_COLS = slice(4, 7)
TRUNK_COLS = slice(7, 10)
ANNOT_COL = 10


def load_subject_runs(dataset_dir: str | Path, subject: int) -> list[pd.DataFrame]:
    """Load every recording run for a given subject as a list of DataFrames."""
    pattern = str(Path(dataset_dir) / f"S{subject:02d}R*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for subject {subject:02d} in {dataset_dir}")
    return [pd.read_csv(f, sep=r"\s+", header=None, dtype=np.float32) for f in files]


def subject_from_filename(path: str | Path) -> int:
    """Extract the subject id from a Daphnet file name like 'S01R01.txt'."""
    m = re.search(r"S(\d+)R", str(path))
    if not m:
        raise ValueError(f"Cannot parse subject from {path}")
    return int(m.group(1))


def concatenate_runs(runs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple runs with a small NaN gap so they remain separate windows."""
    # Insert one NaN row (annot=0) between runs to prevent windows from bleeding across.
    frames = []
    for i, run in enumerate(runs):
        frames.append(run)
        if i < len(runs) - 1:
            gap = pd.DataFrame([[np.nan] * run.shape[1]], columns=run.columns, dtype=np.float32)
            frames.append(gap)
    return pd.concat(frames, ignore_index=True)


def resample_to_100hz(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a DataFrame from 64 Hz to 100 Hz using FFT interpolation.

    The first and last time values are used to build a uniform 100 Hz time base.
    """
    t0, t1 = df.iloc[0, 0], df.iloc[-1, 0]
    # Total duration in seconds
    duration_s = (t1 - t0) / 1000.0
    n_target = int(np.round(duration_s * TARGET_FS)) + 1

    old_t = (df.iloc[:, 0].values - t0) / 1000.0
    new_t = np.linspace(0, duration_s, n_target)

    new = np.zeros((n_target, df.shape[1]), dtype=np.float32)
    new[:, 0] = new_t * 1000.0 + t0  # reconstruct time in ms

    for col in range(1, df.shape[1]):
        col_vals = df.iloc[:, col].values
        # Drop exact NaN gap rows introduced by concatenate_runs; linear interp handles them.
        new[:, col] = signal.resample(col_vals, n_target)
    return pd.DataFrame(new)


def make_windows(
    df: pd.DataFrame,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    target_fs: int = TARGET_FS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract sliding windows of shank and trunk acceleration.

    Returns
    -------
    shank : (N, W, 3) float32, mg
    trunk : (N, W, 3) float32, mg
    y     : (N,) bool, True if any annotation == 2 in the window
    info  : (N, 2) float32, [freeze_index_shank, freeze_index_trunk] averaged over axes
    """
    # Drop the gap row(s) and any fully-NaN rows.
    arr = df.dropna().to_numpy(dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 0, 3)), np.empty((0, 0, 3)), np.empty(0, dtype=bool), np.empty((0, 2))

    shank = arr[:, SHANK_COLS]
    trunk = arr[:, TRUNK_COLS]
    annot = arr[:, ANNOT_COL].astype(np.int32)

    win = int(window_seconds * target_fs)
    step = int(step_seconds * target_fs)

    shank_w, trunk_w, labels, infos = [], [], [], []
    for start in range(0, len(arr) - win + 1, step):
        end = start + win
        seg_annot = annot[start:end]

        # Skip windows where the subject is not in the experiment at all (annot == 0).
        if np.all(seg_annot == 0):
            continue

        is_freeze = np.any(seg_annot == 2)
        shank_w.append(shank[start:end])
        trunk_w.append(trunk[start:end])
        labels.append(is_freeze)

        fi_shank = freeze_index_per_window(shank[start:end], target_fs)
        fi_trunk = freeze_index_per_window(trunk[start:end], target_fs)
        infos.append([fi_shank, fi_trunk])

    if not shank_w:
        return np.empty((0, win, 3)), np.empty((0, win, 3)), np.empty(0, dtype=bool), np.empty((0, 2))

    return (
        np.stack(shank_w, dtype=np.float32),
        np.stack(trunk_w, dtype=np.float32),
        np.array(labels, dtype=bool),
        np.array(infos, dtype=np.float32),
    )


def freeze_index_per_window(acc: np.ndarray, fs: int) -> float:
    """Mean Freeze Index across the 3 accelerometer axes.

    FI = sum(power in 3-8 Hz) / sum(power in 0.5-3 Hz).
    """
    n = acc.shape[0]
    if n < 8:
        return 0.0

    window = np.hanning(n)
    fi_values = []
    for axis in range(acc.shape[1]):
        x = acc[:, axis] - np.mean(acc[:, axis])
        # Compute power spectral density with a stable frequency resolution.
        freqs, psd = signal.periodogram(x * window, fs=fs)
        loco_band = (freqs >= 0.5) & (freqs <= 3.0)
        freeze_band = (freqs >= 3.0) & (freqs <= 8.0)
        loco_power = np.sum(psd[loco_band])
        freeze_power = np.sum(psd[freeze_band])
        if loco_power > 1e-12:
            fi_values.append(freeze_power / loco_power)
        else:
            fi_values.append(0.0)
    return float(np.mean(fi_values))


def load_subject_windows(
    dataset_dir: str | Path,
    subject: int,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> dict:
    """Load, resample, and window one subject's data.

    Returns a dict with keys: subject, shank, trunk, y, fi, and the original df.
    """
    runs = load_subject_runs(dataset_dir, subject)
    df = concatenate_runs([resample_to_100hz(r) for r in runs])
    shank, trunk, y, fi = make_windows(df, window_seconds, step_seconds)
    return {
        "subject": subject,
        "df": df,
        "shank": shank,
        "trunk": trunk,
        "y": y,
        "fi": fi,
    }


def list_subjects(dataset_dir: str | Path) -> list[int]:
    """Return the list of subject numbers present in the dataset."""
    files = sorted(glob.glob(str(Path(dataset_dir) / "S*R*.txt")))
    subjects = {subject_from_filename(f) for f in files}
    return sorted(subjects)
