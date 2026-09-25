"""Central configuration with board-agnostic, general-purpose pinouts.

All pin numbers are listed as simple integers. Change `PLATFORM` and the
relevant pin map to adapt to your actual board (Raspberry Pi, UNO Q MCU,
ESP32, etc.).
"""

from __future__ import annotations

import enum
import os
from pathlib import Path


class Platform(enum.StrEnum):
    SIMULATOR = "simulator"
    RASPBERRY_PI = "pi"
    ESP32 = "esp32"
    UNO_Q_LINUX = "unoq_linux"
    UNO_Q_MCU = "unoq_mcu"


# Which platform are we running on? Default to simulator for safe development.
PLATFORM = Platform(os.getenv("ACTIVESTEP_PLATFORM", "simulator"))

# Project paths
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "final"
DATASET_DIR = REPO_ROOT / "dataset_fog_release" / "dataset"
DB_PATH = REPO_ROOT / "activestep.db"
AUDIO_DIR = REPO_ROOT / "audio"

# Model / signal parameters
SAMPLE_RATE = 100          # Hz
WINDOW_SECONDS = 2.0       # CNN input window
HOP_SECONDS = 0.25         # inference hop
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
N_AXES = 3

# Cue-modality bit masks
VIBRATION = 0b001
LASER = 0b010
AUDIO = 0b100

# Network
UNOQ_WIFI_SSID = os.getenv("ACTIVESTEP_SSID", "ActiveStep")
UNOQ_WIFI_PASS = os.getenv("ACTIVESTEP_PASS", "activestep")
UNOQ_IP = os.getenv("ACTIVESTEP_UNOQ_IP", "192.168.4.1")
ESP32_IP = os.getenv("ACTIVESTEP_ESP32_IP", "192.168.4.2")
ESP32_UDP_PORT = int(os.getenv("ACTIVESTEP_ESP32_PORT", "5006"))
UNOQ_UDP_PORT = int(os.getenv("ACTIVESTEP_UNOQ_PORT", "5005"))
DASHBOARD_PORT = int(os.getenv("ACTIVESTEP_DASHBOARD_PORT", "8000"))
NODE2_HTTP_URL = os.getenv("ACTIVESTEP_NODE2_URL", "").rstrip("/")
DEMO_METRICS = os.getenv("ACTIVESTEP_DEMO_METRICS", "0") == "1"

# ---------------------------------------------------------------------------
# Generic pin maps (change these integers to match your wiring)
# ---------------------------------------------------------------------------

# General-purpose Raspberry Pi BCM pin numbers.
PI_PINS = {
    "shank_sda": 2,        # I2C SDA (fixed on Pi, but kept for reference)
    "shank_scl": 3,        # I2C SCL
    "shank_int": 4,
    "motor_1": 17,         # vibration motor on the shank band
    "laser_1": 27,         # line laser on the shank band
    "sw_true": 22,         # TRUE freeze label
    "sw_false": 23,        # FALSE alarm label
    "status_led": 24,
}

# ESP32 DevKit V1 (30 pin) pin map.
ESP32_PINS = {
    "shank_sda": 21,
    "shank_scl": 22,
    "shank_int": 19,
    "motor_1": 25,
    "laser_1": 26,
    "sw_true": 32,
    "sw_false": 33,
    "status_led": 2,
}

# UNO Q MCU (STM32U585) digital pin map.
UNOQ_MCU_PINS = {
    "trunk_sda": 20,       # SDA on the Qwiic/header
    "trunk_scl": 21,       # SCL
    "motor_2": 5,
    "laser_2": 6,
    "sw_ok": 2,            # "I'm OK" fall cancel
    "status_led": 13,
}

# Audio clip paths
CHECKIN_WAV = AUDIO_DIR / "checkin.wav"
ALARM_WAV = AUDIO_DIR / "alarm.wav"
METRONOME_BPM_MIN = 60
METRONOME_BPM_MAX = 160
METRONOME_RATIO = 1.10   # 10% faster than natural cadence

# Thresholds
DEFAULT_PFOG_THRESHOLD = 0.7
FREEZE_INDEX_THRESHOLD = 2.0
FALL_FREE_FALL_G = 0.5
FALL_IMPACT_G = 2.5
FALL_STILLNESS_MS = 2000


def get_pins(platform: Platform | None = None) -> dict:
    p = platform or PLATFORM
    if p == Platform.RASPBERRY_PI:
        return PI_PINS
    if p == Platform.ESP32:
        return ESP32_PINS
    if p == Platform.UNO_Q_MCU:
        return UNOQ_MCU_PINS
    return {}  # simulator / linux
