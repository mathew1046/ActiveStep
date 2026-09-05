"""Tests for the hardware abstraction layer."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("ACTIVESTEP_PLATFORM", "simulator")

from activestep.config import Platform
from activestep.hardware import MockBackend, get_backend


def test_mock_backend_default():
    b = get_backend("simulator")
    assert isinstance(b, MockBackend)
    b.cues.set(b.cues.state.__class__(vibration=True, laser=False, audio=False))
    sample = b.imu.read()
    assert sample.shape == (6,)


def test_config_pins():
    from activestep.config import get_pins
    assert get_pins(Platform.RASPBERRY_PI)["motor_1"] == 17
    assert get_pins(Platform.ESP32)["motor_1"] == 25
