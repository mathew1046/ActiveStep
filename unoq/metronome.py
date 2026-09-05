"""Adaptive audio metronome for the UNO Q.

Estimates cadence from the ingested shank-IMU stream, synthesizes a click
train at ~1.1x the measured cadence, and plays it to the Bluetooth speaker
via the default audio output (PipeWire/PulseAudio routes A2DP).

Also publishes the current tempo so the ESP32 can pulse vibration in sync.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import numpy as np

try:
    import sounddevice as sd
except OSError:  # no audio device (e.g. dev laptop without ALSA)
    sd = None

from .ingest import STATE

SAMPLE_RATE = 44100
TARGET_RATIO = 1.10  # metronome slightly faster than free cadence (RAC literature)
MIN_BPM, MAX_BPM = 60, 160


def estimate_cadence(imu_windows: list) -> float:
    """Autocorrelation cadence (steps/min) from vertical shank acceleration."""
    if not imu_windows:
        return 0.0
    x = np.asarray([s[1] for s in imu_windows], dtype=np.float32)  # vertical axis
    x = x - x.mean()
    if np.abs(x).max() < 1e-6:
        return 0.0
    r = np.correlate(x, x, mode="full")[len(x) - 1:]
    r = r / (r[0] + 1e-12)
    fs = 100.0
    lo, hi = int(fs * 0.4), int(fs * 1.5)  # 40-150 steps/min lag range
    seg = r[lo:hi]
    if len(seg) == 0 or seg.max() < 0.3:  # no periodic gait
        return 0.0
    period = (np.argmax(seg) + lo) / fs
    return 60.0 / period


def click_track(bpm: float, seconds: float = 30.0) -> np.ndarray:
    """Generate a click track: 10 ms 2 kHz sine blips at `bpm`."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    period = 60.0 / bpm
    phase = np.mod(t, period)
    click = np.where(phase < 0.010, np.sin(2 * np.pi * 2000 * phase), 0.0)
    return click.astype(np.float32)


class Metronome:
    def __init__(self):
        self.cadence = 0.0
        self.bpm = 0.0
        self.playing = False
        self._recent: deque = deque(maxlen=400)  # 4 s @ 100 Hz
        self._stream = None

    async def run(self):
        q = asyncio.Queue(maxsize=100)
        STATE.subscribers.append(q)
        while True:
            msg = await q.get()
            for s in msg.get("imu", []):
                self._recent.append(s)
            cueing = msg.get("state") == "CUEING"
            if cueing and not self.playing:
                self.start()
            elif not cueing and self.playing:
                self.stop()
            if self.playing:
                self._update_tempo()

    def start(self):
        self.cadence = estimate_cadence(list(self._recent))
        base = self.cadence if self.cadence > 40 else 100.0
        self.bpm = float(np.clip(base * TARGET_RATIO, MIN_BPM, MAX_BPM))
        self.playing = True
        if sd is not None:
            track = click_track(self.bpm)
            self._stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            self._stream.start()
            self._loop_track = np.tile(track, 4)  # ~2 min buffer
            self._pos = 0
        print(f"[metronome] start cadence={self.cadence:.1f} bpm={self.bpm:.1f}")

    def _update_tempo(self):
        c = estimate_cadence(list(self._recent))
        if c > 40:
            self.cadence = c
            self.bpm = float(np.clip(c * TARGET_RATIO, MIN_BPM, MAX_BPM))

    def pump(self):
        """Call periodically to feed the audio stream while playing."""
        if not (self.playing and self._stream):
            return
        chunk = self._loop_track[self._pos : self._pos + 2048]
        if len(chunk) < 2048:
            chunk = np.concatenate([chunk, self._loop_track[: 2048 - len(chunk)]])
        self._stream.write(chunk[:, None])
        self._pos = (self._pos + 2048) % len(self._loop_track)

    def stop(self):
        self.playing = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[metronome] stop")


async def audio_pump(metro: Metronome):
    while True:
        metro.pump()
        await asyncio.sleep(0.02)


async def main():
    metro = Metronome()
    await asyncio.gather(metro.run(), audio_pump(metro))


if __name__ == "__main__":
    asyncio.run(main())
