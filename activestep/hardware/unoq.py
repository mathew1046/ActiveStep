"""UNO Q backend using Bridge RPC (Linux side talks to MCU)."""

from __future__ import annotations

import json
import time

import numpy as np

try:
    from arduino_uno_q import Bridge  # placeholder for actual App Lab Bridge module
except ImportError:
    Bridge = None  # type: ignore

from activestep.config import Platform, get_pins
from activestep.hardware.base import (
    CueOutput,
    CueState,
    HardwareBackend,
    IMUReader,
    StatusLED,
    SwitchInput,
)


class UnoQBridgeBackend(HardwareBackend):
    """Talks to the UNO Q MCU over the Arduino Bridge RPC.

    The MCU runs the sketch in `firmware/unoq_app_lab/sketch/` which reads the
    trunk IMU and the "I"m OK" switch and accepts motor/laser commands.
    """

    def __init__(self):
        if Bridge is None:
            raise RuntimeError(
                "Bridge not available. On the UNO Q use Arduino App Lab; "
                "on a dev laptop use ACTIVESTEP_PLATFORM=simulator."
            )
        self.bridge = Bridge()
        self.cues = UnoQCueOutput(self.bridge)
        self.switches = UnoQSwitchInput(self.bridge)
        self.led = UnoQStatusLED(self.bridge)
        self.imu = UnoQIMUReader(self.bridge)

    def tick(self):
        time.sleep(0.001)


class UnoQCueOutput(CueOutput):
    def __init__(self, bridge):
        self._b = bridge

    def set(self, state: CueState):
        mask = (int(state.vibration) | (int(state.laser) << 1) | (int(state.audio) << 2))
        self._b.send(f"CUE {mask}")


class UnoQSwitchInput(SwitchInput):
    def __init__(self, bridge):
        self._b = bridge
        self._cache = {"true": False, "false": False, "ok": False}

    def read(self) -> dict[str, bool]:
        data = self._b.recv_nonblocking()
        if data:
            try:
                d = json.loads(data)
                if "ok" in d:
                    self._cache["ok"] = bool(d["ok"])
            except json.JSONDecodeError:
                pass
        out = self._cache.copy()
        self._cache = {k: False for k in self._cache}
        return out


class UnoQIMUReader(IMUReader):
    def __init__(self, bridge):
        self._b = bridge
        self._last = np.zeros(6, dtype=np.float32)

    def read(self) -> np.ndarray:
        data = self._b.recv_nonblocking()
        if data and data.startswith("IMU "):
            try:
                parts = data.split()
                vals = [float(x) for x in parts[2:8]]
                self._last = np.array(vals, dtype=np.float32)
            except Exception:
                pass
        return self._last


class UnoQStatusLED(StatusLED):
    def __init__(self, bridge):
        self._b = bridge

    def set(self, on: bool):
        self._b.send(f"LED {1 if on else 0}")
