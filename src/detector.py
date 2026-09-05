"""Shared Tier-1 FOG detector core.

One implementation of the scoring and cue-state logic, used by:
- the device runtime (activestep/runner.py, sample-push mode via StreamDetector)
- the offline streaming benchmark (src/benchmark.py, batch mode via CueFSM)
- firmware parity checks (same weights, thresholds, and hysteresis as the ESP32)

Score convention (must match src/features.p_fog_combined):
    score = w_cnn * clip(p_cnn, 0, 1) + w_fi * clip((fi - fi_threshold)/(2*fi_threshold) + 0.5, 0, 1)
Freeze Index is computed on RAW mg windows (scale-invariant ratio, physically
meaningful); the learned model sees scaler-normalized windows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from src.features import compute_freeze_index, p_fog_combined
except ImportError:  # imported as a top-level module
    from features import compute_freeze_index, p_fog_combined


@dataclass
class DetectorConfig:
    window_samples: int = 200
    hop_samples: int = 25
    w_cnn: float = 0.6
    w_fi: float = 0.4
    fi_threshold: float = 2.0
    cnn_threshold: float = 0.5
    threshold: float = 0.7   # cue ON threshold for the combined score
    hysteresis: float = 0.15 # cue OFF at threshold - hysteresis


def combine_scores(p_cnn, fi, cfg: DetectorConfig):
    """Combined pFOG score; identical math to features.p_fog_combined."""
    return p_fog_combined(
        p_cnn, fi,
        w_cnn=cfg.w_cnn, w_fi=cfg.w_fi,
        fi_threshold=cfg.fi_threshold, cnn_threshold=cfg.cnn_threshold,
    )


@dataclass
class CueEvent:
    type: str                 # "cue_start" | "cue_stop"
    t_ms: int
    score: float | None
    recovery_ms: int | None = None  # cue duration, set on cue_stop
    censored: bool = False          # cue_stop forced at stream end


class CueFSM:
    """Hysteresis cue state machine over timestamped scores.

    Cue turns ON when score >= threshold, OFF when score < threshold - hysteresis.
    """

    def __init__(self, threshold: float, hysteresis: float):
        self.threshold = float(threshold)
        self.hysteresis = float(hysteresis)
        self.cueing = False
        self.cue_start_ms: int | None = None
        self.last_t_ms: int | None = None

    def push(self, t_ms: int, score: float) -> list[CueEvent]:
        events: list[CueEvent] = []
        if not self.cueing and score >= self.threshold:
            self.cueing = True
            self.cue_start_ms = int(t_ms)
            events.append(CueEvent("cue_start", int(t_ms), float(score)))
        elif self.cueing and score < self.threshold - self.hysteresis:
            self.cueing = False
            events.append(
                CueEvent(
                    "cue_stop", int(t_ms), float(score),
                    recovery_ms=int(t_ms) - int(self.cue_start_ms),
                )
            )
            self.cue_start_ms = None
        self.last_t_ms = int(t_ms)
        return events

    def finish(self, t_end_ms: int | None = None) -> list[CueEvent]:
        """Close an ongoing cue at stream end (recorded as censored)."""
        if not self.cueing:
            return []
        t = int(t_end_ms if t_end_ms is not None else self.last_t_ms)
        self.cueing = False
        ev = CueEvent(
            "cue_stop", t, None,
            recovery_ms=t - int(self.cue_start_ms),
            censored=True,
        )
        self.cue_start_ms = None
        return [ev]

    def reset(self) -> None:
        self.cueing = False
        self.cue_start_ms = None
        self.last_t_ms = None


@dataclass
class Decision:
    t_ms: int
    fi: float
    p_cnn: float | None
    score: float
    cueing: bool
    events: list[CueEvent] = field(default_factory=list)


class StreamDetector:
    """Sample-push detector: ring buffer -> FI (+ optional model) -> score -> cue FSM.

    `predict_fn` contract: (N, W, 3) float32 normalized windows -> (N,) float
    probabilities. If None, the detector is Freeze-Index-only (set w_cnn=0).
    `scaler` must expose .transform((W,3)) and is applied before predict_fn.
    """

    def __init__(
        self,
        config: DetectorConfig,
        predict_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        scaler=None,
        threshold_offset: float = 0.0,
    ):
        self.cfg = config
        self.predict_fn = predict_fn
        self.scaler = scaler
        self.threshold_offset = float(threshold_offset)
        self._ring: deque[np.ndarray] = deque(maxlen=config.window_samples)
        self._since_last = 0
        self._fsm = CueFSM(self.effective_threshold, config.hysteresis)
        self._window = np.zeros((config.window_samples, 3), dtype=np.float32)

    @property
    def effective_threshold(self) -> float:
        return self.cfg.threshold + self.threshold_offset

    def set_threshold_offset(self, offset: float) -> None:
        self.threshold_offset = float(offset)
        self._fsm.threshold = self.effective_threshold

    def push(self, t_ms: int, acc3: np.ndarray) -> Decision | None:
        """Feed one 3-axis sample (mg); returns a Decision every hop, else None."""
        self._ring.append(np.asarray(acc3, dtype=np.float32)[:3])
        self._since_last += 1
        if len(self._ring) < self.cfg.window_samples or self._since_last < self.cfg.hop_samples:
            return None
        self._since_last = 0

        win = np.stack(self._ring, axis=0)  # (W, 3) raw mg
        fi = float(compute_freeze_index(win[None, ...], fs=100)[0])

        p_cnn = None
        if self.predict_fn is not None:
            x = self.scaler.transform(win) if self.scaler is not None else win
            p_cnn = float(np.asarray(self.predict_fn(x[None, ...].astype(np.float32))).ravel()[0])

        score = float(combine_scores(p_cnn if p_cnn is not None else 0.0, fi, self.cfg))
        self._fsm.threshold = self.effective_threshold
        events = self._fsm.push(int(t_ms), score)
        return Decision(
            t_ms=int(t_ms), fi=fi, p_cnn=p_cnn, score=score,
            cueing=self._fsm.cueing, events=events,
        )

    def finish(self, t_end_ms: int | None = None) -> list[CueEvent]:
        return self._fsm.finish(t_end_ms)

    def recent(self, n: int) -> np.ndarray:
        """Last `n` raw samples (for telemetry)."""
        items = list(self._ring)[-n:]
        return np.stack(items, axis=0) if items else np.empty((0, 3), np.float32)
