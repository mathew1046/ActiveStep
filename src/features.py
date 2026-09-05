"""Feature extraction for both the ESP32 (Tier-1) and the UNO Q (Tier-2)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def normalize_shank_window(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score a single (W, 3) or (N, W, 3) shank window."""
    return (window - mean) / (std + 1e-8)


def compute_freeze_index(
    acc: np.ndarray,
    fs: int = 100,
    freeze_band: tuple[float, float] = (3.0, 8.0),
    locomotor_band: tuple[float, float] = (0.5, 3.0),
) -> np.ndarray:
    """Vectorized Freeze Index across the last axis.

    Parameters
    ----------
    acc : (..., T, 3) array of mg

    Returns
    -------
    fi : (...) array
    """
    original_shape = acc.shape[:-2]
    n = acc.shape[-2]
    n_axes = acc.shape[-1]
    flat = acc.reshape(-1, n, n_axes)
    out = np.zeros(flat.shape[0], dtype=np.float32)

    if n < 8:
        return np.zeros(original_shape, dtype=np.float32)

    w = np.hanning(n)
    for i, win in enumerate(flat):
        fi_axes = []
        for ax in range(n_axes):
            x = win[:, ax] - win[:, ax].mean()
            freqs, psd = signal.periodogram(x * w, fs=fs)
            loco = (freqs >= locomotor_band[0]) & (freqs <= locomotor_band[1])
            freeze = (freqs >= freeze_band[0]) & (freqs <= freeze_band[1])
            lp = psd[loco].sum()
            fp = psd[freeze].sum()
            fi_axes.append(fp / (lp + 1e-12))
        out[i] = float(np.mean(fi_axes))

    return out.reshape(original_shape)


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
    """Compact streaming-safe feature vector from one (T, 3) raw-mg window.

    Used by the classical (logistic) baseline candidate. All features are
    computable causally from a single trailing window — no future samples,
    no cross-window statistics.
    """
    n, n_axes = acc.shape
    feats = []
    if n < 8:
        return np.zeros(12, dtype=np.float32)
    w = np.hanning(n)
    fi_ax, loco_tot, freeze_tot = [], 0.0, 0.0
    for ax in range(n_axes):
        x = acc[:, ax] - acc[:, ax].mean()
        freqs, psd = signal.periodogram(x * w, fs=fs)
        loco = (freqs >= locomotor_band[0]) & (freqs <= locomotor_band[1])
        freeze = (freqs >= freeze_band[0]) & (freqs <= freeze_band[1])
        lp, fp = float(psd[loco].sum()), float(psd[freeze].sum())
        fi_ax.append(fp / (lp + 1e-12))
        loco_tot += lp
        freeze_tot += fp
    feats.append(float(np.mean(fi_ax)))
    feats.extend(fi_ax)
    feats.append(np.log10(loco_tot + 1e-6))
    feats.append(np.log10(freeze_tot + 1e-6))
    feats.extend(acc.std(axis=0).tolist())
    feats.extend((acc.max(axis=0) - acc.min(axis=0)).tolist())
    return np.asarray(feats, dtype=np.float32)


def window_feature_matrix(X: np.ndarray, fs: int = 100) -> np.ndarray:
    """Stack window_feature_vector over an (N, T, 3) batch -> (N, 12)."""
    return np.stack([window_feature_vector(x, fs=fs) for x in X], axis=0)
