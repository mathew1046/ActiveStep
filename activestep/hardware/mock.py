"""Mock / simulator backend. Replays Daphnet data and fakes cues/switches."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from activestep.config import DATASET_DIR, get_pins
from activestep.hardware.base import (
    CueOutput,
    CueState,
    HardwareBackend,
    IMUReader,
    StatusLED,
    SwitchInput,
)


class MockCueOutput(CueOutput):
    def __init__(self):
        self.state = CueState()

    def set(self, state: CueState):
        if state != self.state:
            print(f"[mock] cue: vib={state.vibration} laser={state.laser} audio={state.audio}")
            self.state = state


class MockSwitchInput(SwitchInput):
    def __init__(self):
        self.presses = {"true": False, "false": False, "ok": False}

    def set(self, key: str, value: bool = True):
        self.presses[key] = value

    def read(self) -> dict[str, bool]:
        out = self.presses.copy()
        self.presses = {k: False for k in self.presses}
        return out


class MockStatusLED(StatusLED):
    def set(self, on: bool):
        print(f"[mock] status LED {'ON' if on else 'OFF'}")


class ReplayIMUReader(IMUReader):
    """Replays a Daphnet subject at 100 Hz for simulation."""

    def __init__(self, subject: int = 1, dataset_dir: Path | None = None):
        from src.data import load_subject_windows

        d = load_subject_windows(dataset_dir or DATASET_DIR, subject)
        self.windows = d["shank"]
        self.labels = d["y"]
        self.w = 0
        self.i = 0
        self.t0 = time.time()

    def read(self) -> np.ndarray:
        target_i = int((time.time() - self.t0) * 100)
        self.i = min(target_i, self.windows.shape[1] - 1)
        if self.w >= len(self.windows):
            self.w = 0
            self.t0 = time.time()
            self.i = 0
        if self.i >= self.windows.shape[1]:
            self.w = (self.w + 1) % len(self.windows)
            self.i = 0
            self.t0 = time.time()
        sample = np.zeros(6, dtype=np.float32)
        sample[:3] = self.windows[self.w, self.i]
        return sample


class MockIMUReader(IMUReader):
    """Returns a small synthetic walking signal."""

    def __init__(self, fs: int = 100):
        self.fs = fs
        self.i = 0

    def read(self) -> np.ndarray:
        t = self.i / self.fs
        self.i += 1
        x = np.sin(2 * np.pi * 1.5 * t) * 300
        y = 1000 + np.sin(2 * np.pi * 1.5 * t + np.pi / 2) * 400
        z = np.cos(2 * np.pi * 1.5 * t) * 200
        return np.array([x, y, z, 0, 0, 0], dtype=np.float32)


class MockBackend(HardwareBackend):
    """Complete simulator with configurable fake IMU and replay."""

    def __init__(self, replay: bool = True, subject: int = 1):
        self.cues = MockCueOutput()
        self.switches = MockSwitchInput()
        self.led = MockStatusLED()
        self.imu = ReplayIMUReader(subject) if replay else MockIMUReader()
        self._pins = get_pins()

    def tick(self):
        pass
