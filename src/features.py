"""Feature extraction for both the ESP32 (Tier-1) and the UNO Q (Tier-2)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def normalize_shank_window(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score a single (W, 3) or (N, W, 3) shank window."""
    return (window - mean) / (std + 1e-8)


def freeze_band_powers(
    acc: np.ndarray,
    fs: int = 100,
    freeze_band: tuple[float, float] = (3.0, 8.0),
    locomotor_band: tuple[float, float] = (0.5, 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return locomotor- and freeze-band powers for every input axis."""
    acc = np.asarray(acc)
    original_shape = acc.shape[:-2]
    n, n_axes = acc.shape[-2:]
    if n < 8:
        shape = (*original_shape, n_axes)
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    flat = acc.reshape(-1, n, n_axes).astype(np.float64, copy=False)
    centered = flat - flat.mean(axis=1, keepdims=True)
    freqs, psd = signal.periodogram(centered * np.hanning(n)[None, :, None], fs=fs, axis=1)
    locomotor = (freqs >= locomotor_band[0]) & (freqs < locomotor_band[1])
    freeze = (freqs >= freeze_band[0]) & (freqs <= freeze_band[1])
    loco_power = psd[:, locomotor].sum(axis=1)
    freeze_power = psd[:, freeze].sum(axis=1)
    shape = (*original_shape, n_axes)
    return loco_power.reshape(shape).astype(np.float32), freeze_power.reshape(shape).astype(np.float32)


def freeze_index_components(
    acc: np.ndarray,
    fs: int = 100,
    freeze_band: tuple[float, float] = (3.0, 8.0),
    locomotor_band: tuple[float, float] = (0.5, 3.0),
    aggregate_axes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable Freeze Index and its absolute 0.5–8 Hz power."""
    loco, freeze = freeze_band_powers(acc, fs, freeze_band, locomotor_band)
    total_power = (loco + freeze).sum(axis=-1)
    if aggregate_axes:
        fi = freeze.sum(axis=-1) / (loco.sum(axis=-1) + 1e-12)
    else:
        fi = np.mean(freeze / (loco + 1e-12), axis=-1)
    return np.asarray(fi, np.float32), np.asarray(total_power, np.float32)


def compute_freeze_index(
    acc: np.ndarray,
    fs: int = 100,
    freeze_band: tuple[float, float] = (3.0, 8.0),
    locomotor_band: tuple[float, float] = (0.5, 3.0),
    aggregate_axes: bool = False,
) -> np.ndarray:
    """Compute the Freeze Index for one or more raw-mg windows."""
    fi, _ = freeze_index_components(acc, fs, freeze_band, locomotor_band, aggregate_axes)
    return fi


def compute_power_gated_freeze_index(
    acc: np.ndarray,
    power_threshold: float,
    fs: int = 100,
) -> np.ndarray:
    """Compute aggregate-axis FI, suppressing windows below an absolute power floor."""
    fi, power = freeze_index_components(acc, fs=fs, aggregate_axes=True)
    return np.where(power >= power_threshold, fi, 0.0).astype(np.float32)


def compute_cadence(acc_vertical: np.ndarray, fs: int = 100) -> float:
    """Estimate stride cadence (steps/min) from the vertical shank acceleration.

    Uses autocorrelation with gravity removed and a lag window corresponding to
    30–120 steps per minute.
    """
    x = acc_vertical - np.mean(acc_vertical)
    # Limit lag to 0.5 s (2 Hz / 120 steps/min) up to 2 s (0.5 Hz / 30 steps/min)
    min_lag = int(fs * 0.5)
    max_lag = min(int(fs * 2.0), len(x) - 1)
    if max_lag <= min_lag:
        return 0.0

    # Autocorrelation
    xn = x - x.mean()
    r = np.correlate(xn, xn, mode="full")[len(xn) - 1:]
    r = r / (r[0] + 1e-12)
    segment = r[min_lag:max_lag]
    if len(segment) == 0:
        return 0.0
    peak_lag = np.argmax(segment) + min_lag
    period_s = peak_lag / fs
    if period_s <= 0:
        return 0.0
    return 60.0 / period_s


def step_time_asymmetry(left: np.ndarray, right: np.ndarray, fs: int = 100) -> float:
    """Placeholder for bilateral asymmetry.

    With only a single IMU we can estimate inter-step interval variance; with
    two leg sensors this becomes the absolute difference in step times.
    """
    cad_l = compute_cadence(left, fs)
    cad_r = compute_cadence(right, fs)
    if cad_l + cad_r == 0:
        return 0.0
    return float(2 * abs(cad_l - cad_r) / (cad_l + cad_r + 1e-12))


def festination_index(
    cadence: float,
    step_amplitude: float,
    baseline_cadence: float,
    baseline_amplitude: float,
) -> float:
    """Festination = cadence rising while step amplitude falls.

    Returns a positive value when cadence is higher than baseline and amplitude
    is lower than baseline, typical of festinating gait before a freeze.
    """
    if baseline_cadence <= 0 or baseline_amplitude <= 0:
        return 0.0
    cad_ratio = cadence / baseline_cadence
    amp_ratio = step_amplitude / (baseline_amplitude + 1e-12)
    # High cadence * low amplitude -> high festination.
    return float(max(0.0, cad_ratio - amp_ratio))


def step_amplitude(acc: np.ndarray) -> float:
    """Peak-to-peak vertical acceleration amplitude, mg."""
    return float(np.max(acc) - np.min(acc))


def turning_rate(gyro_yaw: np.ndarray, fs: int = 100, window_s: float = 2.0) -> float:
    """Integrated yaw gyro over the last window (degrees)."""
    # Daphnet has only accelerometers. On the UNO Q the MPU-6050 gives gyro.
    n = int(window_s * fs)
    seg = gyro_yaw[-n:] if len(gyro_yaw) >= n else gyro_yaw
    return float(np.trapz(seg, dx=1.0 / fs) * (180.0 / np.pi))


def p_fog_combined(
    p_cnn: np.ndarray | float,
    fi: np.ndarray | float,
    w_cnn: float = 0.6,
    w_fi: float = 0.4,
    fi_threshold: float = 2.0,
    cnn_threshold: float = 0.5,
) -> np.ndarray | float:
    """Combine a learned pFOG with the Freeze Index into one score in [0, 1].

    Both sources are calibrated into [0, 1] before weighting so the final
    score is explainable and does not collapse when one source is noisy.
    (Moved from src/model.py so the streaming core has no TensorFlow import;
    src/model.py re-exports it for compatibility.)
    """
    fi_score = np.clip((fi - fi_threshold) / (fi_threshold * 2.0 + 1e-8) + 0.5, 0, 1)
    cnn_score = np.clip(np.asarray(p_cnn, dtype=float) / (cnn_threshold * 2.0 + 1e-8), 0, 1)
    return w_cnn * cnn_score + w_fi * fi_score


def window_feature_vector(
    acc: np.ndarray,
    fs: int = 100,
    freeze_band: tuple[float, float] = (3.0, 8.0),
    locomotor_band: tuple[float, float] = (0.5, 3.0),
) -> np.ndarray:
    """Compact streaming-safe feature vector from one raw-mg window."""
    return window_feature_matrix(
        np.asarray(acc, dtype=np.float32)[None], fs, freeze_band, locomotor_band
    )[0]


def window_feature_matrix(
    X: np.ndarray,
    fs: int = 100,
    freeze_band: tuple[float, float] = (3.0, 8.0),
    locomotor_band: tuple[float, float] = (0.5, 3.0),
) -> np.ndarray:
    """Extract bounded spectral and time-domain features from trailing windows."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.empty((0, 12), dtype=np.float32)
    if X.shape[1] < 8:
        return np.zeros((len(X), 12), dtype=np.float32)
    loco, freeze = freeze_band_powers(X, fs, freeze_band, locomotor_band)
    axis_fi = np.log1p(freeze / (loco + 1e-12))
    aggregate_fi = np.log1p(freeze.sum(axis=1) / (loco.sum(axis=1) + 1e-12))[:, None]
    powers = np.stack(
        [np.log10(loco.sum(axis=1) + 1e-6), np.log10(freeze.sum(axis=1) + 1e-6)],
        axis=1,
    )
    return np.concatenate(
        [aggregate_fi, axis_fi, powers, X.std(axis=1), np.ptp(X, axis=1)], axis=1
    ).astype(np.float32)
