"""Hardware abstraction layer: cue outputs, switches, IMU, status LED."""

from __future__ import annotations

from .base import CueOutput, SwitchInput, IMUReader, StatusLED, HardwareBackend
from .mock import MockBackend

try:
    from .pi import PiBackend
except ImportError:
    PiBackend = None  # type: ignore

try:
    from .unoq import UnoQBridgeBackend
except ImportError:
    UnoQBridgeBackend = None  # type: ignore


__all__ = [
    "CueOutput",
    "SwitchInput",
    "IMUReader",
    "StatusLED",
    "HardwareBackend",
    "MockBackend",
    "PiBackend",
    "UnoQBridgeBackend",
    "get_backend",
]


def get_backend(name: str | None = None) -> HardwareBackend:
    """Return the configured backend.

    name='mock' | 'pi' | 'unoq' | None (auto from ACTIVESTEP_PLATFORM)
    """
    from activestep.config import Platform, PLATFORM

    n = (name or PLATFORM.value).lower()
    if n == "pi" or n == "raspberry_pi":
        if PiBackend is None:
            raise RuntimeError("Pi backend not available; install gpiozero")
        return PiBackend()
    if n == "unoq" or n.startswith("unoq"):
        if UnoQBridgeBackend is None:
            raise RuntimeError("UNO Q backend not available; install pyserial")
        return UnoQBridgeBackend()
    return MockBackend()
