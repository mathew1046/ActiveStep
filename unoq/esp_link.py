"""Command channel from the UNO Q Linux side to the ESP32.

ESP32 listens on UDP (default 5006). Commands are JSON:
{"threshold_offset": -0.05}
{"modality_mask": 5}            # bitmask: bit0=vib, bit1=laser, bit2=audio
{"tempo_bpm": 104.2}            # metronome tempo for haptic sync
{"model_ota": "base64..."}      # optional future OTA
"""

from __future__ import annotations

import json
import socket

ESP32_IP = "192.168.4.2"  # ESP32 STA IP after joining the UNO Q AP
ESP32_PORT = 5006


_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def _send(cmd: dict):
    try:
        _sock.sendto(json.dumps(cmd).encode(), (ESP32_IP, ESP32_PORT))
    except OSError:
        pass


def send_threshold_offset(offset: float):
    _send({"threshold_offset": round(offset, 3)})


def send_modality_mask(mask: int):
    _send({"modality_mask": int(mask)})


def send_tempo(bpm: float):
    _send({"tempo_bpm": round(bpm, 1)})
