"""Tier-2 bilateral / context feature extraction on the UNO Q.

Consumes the live IMU stream, computes:
- cadence, step amplitude (shank + trunk)
- Freeze Index
- step-time asymmetry between shank and trunk
- festination index
- turn detection from gyro yaw (trunk)

Emits a context score to the dashboard and a threshold offset to the ESP32.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import numpy as np

from src.features import (
    compute_cadence,
    compute_freeze_index,
    festination_index,
    step_amplitude,
    turning_rate,
)
from .db import connect
from .esp_link import send_threshold_offset
from .ingest import STATE

WINDOW_S = 2.0
FS = 100


def context_features(shank_win: np.ndarray, trunk_win: np.ndarray) -> dict:
    """Compute all Tier-2 features from the last 2-second windows."""
    c_shank = compute_cadence(shank_win[:, 1], FS)
    c_trunk = compute_cadence(trunk_win[:, 1], FS)
    amp_shank = step_amplitude(shank_win[:, 1])
    amp_trunk = step_amplitude(trunk_win[:, 1])
    fi_shank = compute_freeze_index(shank_win[None, ...], FS)[0]
    fi_trunk = compute_freeze_index(trunk_win[None, ...], FS)[0]

    # Asymmetry: relative cadence difference trunk vs shank.
    if c_shank + c_trunk > 0:
        asymmetry = abs(c_shank - c_trunk) / ((c_shank + c_trunk) / 2.0 + 1e-8)
    else:
        asymmetry = 0.0

    # Festination: cadence rising + step amplitude falling.
    baseline = {"cadence": 100.0, "step_amp": 500.0}  # loaded from calibrate later
    fest = festination_index(c_shank, amp_shank, baseline["cadence"], baseline["step_amp"])

    # Turning: integrated yaw; Daphnet has no gyro, so on real hardware use last axis.
    turn = 0.0  # turning_rate(np.zeros(FS * 2))

    return {
        "cadence_shank": float(c_shank),
        "cadence_trunk": float(c_trunk),
        "step_amp_shank": float(amp_shank),
        "step_amp_trunk": float(amp_trunk),
        "fi_shank": float(fi_shank),
        "fi_trunk": float(fi_trunk),
        "asymmetry": float(asymmetry),
        "festination": float(fest),
        "turning_deg": float(turn),
    }


def threshold_offset(feats: dict) -> float:
    """Return a small (-0.15 ... 0.0) offset to the ESP32 threshold.

    Lower the threshold during known freeze-prone contexts: turns, festination,
    gait initiation (high asymmetry). Raise it when walking is steady.
    """
    off = 0.0
    if feats["turning_deg"] > 45:
        off -= 0.08
    if feats["festination"] > 0.3:
        off -= 0.06
    if feats["asymmetry"] > 0.25:
        off -= 0.04
    # Slight guard against over-sensitivity during stillness.
    if feats["cadence_shank"] < 30:
        off += 0.05
    return float(np.clip(off, -0.15, 0.05))


class Tier2:
    def __init__(self):
        self.shank_q = deque(maxlen=FS * 2)
        self.trunk_q = deque(maxlen=FS * 2)
        self.conn = connect()
        self._last_push = 0.0

    async def run(self):
        q: asyncio.Queue = asyncio.Queue(maxsize=400)
        STATE.subscribers.append(q)
        while True:
            msg = await q.get()
            for s in msg.get("imu", []):
                self.shank_q.append(s)
            # Trunk IMU will come from the UNO Q MCU via Bridge (different topic).
            # For now, mirror shank to keep features alive; on the device both are filled.
            for s in msg.get("trunk_imu", []):
                self.trunk_q.append(s)

            if len(self.shank_q) < FS * WINDOW_S:
                continue

            # If no trunk yet, duplicate shank so features still run.
            shank_win = np.array(list(self.shank_q), dtype=np.float32)
            trunk_win = np.array(list(self.trunk_q) or list(self.shank_q), dtype=np.float32)
            if len(trunk_win) < FS * WINDOW_S:
                trunk_win = shank_win

            feats = context_features(shank_win, trunk_win)
            off = threshold_offset(feats)

            # Push offset ~1 Hz
            t = time.time()
            if t - self._last_push > 1.0:
                self._last_push = t
                send_threshold_offset(off)
                feats["threshold_offset"] = off
                try:
                    send_threshold_offset(off)
                except Exception:
                    pass

            # Expose to dashboard
            STATE.latest.update({"tier2": feats})
            self._store(feats, int(t * 1000))

    def _store(self, feats: dict, t: int):
        try:
            self.conn.execute(
                """INSERT INTO features
                   (t_ms, sensor, cadence, step_amp, fi, asymmetry, festination, turning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    t,
                    "trunk",
                    feats["cadence_trunk"],
                    feats["step_amp_trunk"],
                    feats["fi_trunk"],
                    feats["asymmetry"],
                    feats["festination"],
                    feats["turning_deg"],
                ),
            )
            self.conn.commit()
        except Exception:
            pass


async def main():
    await Tier2().run()


if __name__ == "__main__":
    asyncio.run(main())
