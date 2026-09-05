"""Raspberry Pi GPIO backend using gpiozero (general Pi pinouts)."""

from __future__ import annotations

import time

import numpy as np

try:
    from gpiozero import Button, DigitalOutputDevice
    from smbus2 import SMBus
except ImportError as e:
    raise ImportError("Pi backend requires 'gpiozero' and 'smbus2'.") from e

from activestep.config import Platform, get_pins
from activestep.hardware.base import (
    CueOutput,
    CueState,
    HardwareBackend,
    IMUReader,
    StatusLED,
    SwitchInput,
)


class PiCueOutput(CueOutput):
    """Low-side switched motor and laser."""

    def __init__(self, pins: dict):
        self.motor = DigitalOutputDevice(pins["motor_1"])
        self.laser = DigitalOutputDevice(pins["laser_1"])

    def set(self, state: CueState):
        self.motor.value = state.vibration
        self.laser.value = state.laser


class PiSwitchInput(SwitchInput):
    def __init__(self, pins: dict):
        self.sw_true = Button(pins["sw_true"], pull_up=True)
        self.sw_false = Button(pins["sw_false"], pull_up=True)
        self.sw_ok = Button(pins.get("sw_ok", 26), pull_up=True)

    def read(self) -> dict[str, bool]:
        return {
            "true": self.sw_true.is_pressed,
            "false": self.sw_false.is_pressed,
            "ok": self.sw_ok.is_pressed,
        }


class PiStatusLED(StatusLED):
    def __init__(self, pin: int):
        self.led = DigitalOutputDevice(pin)

    def set(self, on: bool):
        self.led.value = on


class PiIMUReader(IMUReader):
    """MPU-6050 on the Pi I2C bus."""

    def __init__(self, bus: int = 1, addr: int = 0x68):
        self.addr = addr
        self.bus = SMBus(bus)
        self._init_mpu()

    def _init_mpu(self):
        self.bus.write_byte_data(self.addr, 0x6B, 0x00)  # wake
        self.bus.write_byte_data(self.addr, 0x1C, 0x08)  # ±4g
        self.bus.write_byte_data(self.addr, 0x1B, 0x10)  # ±1000 deg/s

    def read(self) -> np.ndarray:
        data = self.bus.read_i2c_block_data(self.addr, 0x3B, 14)
        ax = (data[0] << 8) | data[1]
        ay = (data[2] << 8) | data[3]
        az = (data[4] << 8) | data[5]
        t = (data[6] << 8) | data[7]
        gx = (data[8] << 8) | data[9]
        gy = (data[10] << 8) | data[11]
        gz = (data[12] << 8) | data[13]
        # convert 16-bit signed
        def s16(v):
            return v - 65536 if v > 32767 else v
        gain_a = 1000.0 / 8192.0  # mg
        gain_g = 1000.0 / 32.8     # mdps
        return np.array([
            s16(ax) * gain_a, s16(ay) * gain_a, s16(az) * gain_a,
            s16(gx) * gain_g, s16(gy) * gain_g, s16(gz) * gain_g,
        ], dtype=np.float32)


class PiBackend(HardwareBackend):
    def __init__(self):
        pins = get_pins(Platform.RASPBERRY_PI)
        self.cues = PiCueOutput(pins)
        self.switches = PiSwitchInput(pins)
        self.imu = PiIMUReader()
        self.led = PiStatusLED(pins["status_led"])

    def tick(self):
        time.sleep(0.001)
