"""Abstract base classes for the hardware layer."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CueState:
    vibration: bool = False
    laser: bool = False
    audio: bool = False


class CueOutput(ABC):
    """Controls the cue actuators (motor, laser, audio request)."""

    @abstractmethod
    def set(self, state: CueState):
        pass


class SwitchInput(ABC):
    """Reads debounced microswitches."""

    @abstractmethod
    def read(self) -> dict[str, bool]:
        """Return {'true': ..., 'false': ..., 'ok': ...} presses since last call."""
        pass


class IMUReader(ABC):
    """Reads (accelerometer, gyroscope) samples."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return one sample: [ax, ay, az, gx, gy, gz]."""
        pass


class StatusLED(ABC):
    @abstractmethod
    def set(self, on: bool):
        pass


class HardwareBackend(ABC):
    """Aggregate hardware interface used by the runtime loop."""

    cues: CueOutput
    switches: SwitchInput
    imu: IMUReader
    led: StatusLED

    @abstractmethod
    def tick(self):
        """Optional per-loop hook (audio pump, network polling, etc.)."""
        pass


# ---------------------------------------------------------------------------
# A generic software-debounced switch helper
# ---------------------------------------------------------------------------

class DebouncedSwitch:
    def __init__(self, read_fn: Callable[[], bool], threshold_ms: int = 50):
        self._read = read_fn
        self._threshold_ms = threshold_ms
        self._pressed = False
        self._last_time = 0

    def check(self) -> bool:
        now = time.ticks_ms() if hasattr(time, "ticks_ms") else int(time.monotonic() * 1000)
        raw = self._read()
        if raw and not self._pressed:
            if now - self._last_time > self._threshold_ms:
                self._pressed = True
                self._last_time = now
                return True
        elif not raw:
            self._pressed = False
        return False
