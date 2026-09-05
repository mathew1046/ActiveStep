"""Fall escalation FSM on the UNO Q.

Trigger comes from the STM32 MCU (free-fall -> impact -> stillness on IMU-B)
delivered via Bridge RPC or an ingest packet with {"fall_suspect": true}.

FSM: SUSPECT -> speech check-in -> 20 s countdown -> OK | ALARM
- OK: patient pressed the "I'm OK" microswitch (MCU sends {"fall_ok": true})
- ALARM: speaker alarm + dashboard alert; ntfy/Telegram only if internet is up
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from enum import Enum

try:
    import sounddevice as sd
    import numpy as np
except OSError:
    sd = None
    np = None

from .db import connect
from .ingest import STATE

CHECKIN_CLIP = "audio/checkin.wav"      # "Are you okay? Press your button."
ALARM_CLIP = "audio/alarm.wav"
COUNTDOWN_S = 20


class FallState(Enum):
    IDLE = 0
    CHECKIN = 1
    ALARM = 2


def _play(path: str):
    """Play a WAV file; fall back to a synthesized beep if missing."""
    try:
        import soundfile  # optional
    except ImportError:
        soundfile = None
    if sd is not None and np is not None:
        try:
            import soundfile as sf
            data, fs = sf.read(path, dtype="float32")
            sd.play(data, fs)
            return
        except Exception:
            pass
        # Fallback beep
        t = np.arange(int(44100 * 0.8)) / 44100
        sd.play(0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32), 44100)


def _notify(title: str, body: str):
    """Internet notification — opt-in, only if reachable."""
    try:
        subprocess.run(
            ["curl", "-s", "-m", "3", "-d", body, "https://ntfy.sh/activestep"],
            check=False, timeout=4,
        )
    except Exception:
        pass  # offline: dashboard banner is the alert


class FallFSM:
    def __init__(self):
        self.state = FallState.IDLE
        self._t0 = 0.0
        self.conn = connect()

    async def run(self):
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        STATE.subscribers.append(q)
        while True:
            if self.state == FallState.CHECKIN and time.time() - self._t0 > COUNTDOWN_S:
                self._alarm()
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if msg.get("fall_suspect") and self.state == FallState.IDLE:
                self._checkin()
            elif msg.get("fall_ok") and self.state == FallState.CHECKIN:
                self._ok()

    def _checkin(self):
        self.state = FallState.CHECKIN
        self._t0 = time.time()
        _play(CHECKIN_CLIP)
        print("[fall] check-in started")

    def _ok(self):
        self.state = FallState.IDLE
        self.conn.execute(
            "INSERT INTO falls (t_ms, outcome, response_ms) VALUES (?,?,?)",
            (int(self._t0 * 1000), "OK", int((time.time() - self._t0) * 1000)),
        )
        self.conn.commit()
        print("[fall] cancelled by patient")

    def _alarm(self):
        self.state = FallState.IDLE
        _play(ALARM_CLIP)
        self.conn.execute(
            "INSERT INTO falls (t_ms, outcome, response_ms) VALUES (?,?,?)",
            (int(self._t0 * 1000), "ALARM", COUNTDOWN_S * 1000),
        )
        self.conn.commit()
        _notify("ActiveStep fall alert", "Suspected fall, no response within 20 s")
        print("[fall] ALARM raised")


async def main():
    await FallFSM().run()


if __name__ == "__main__":
    asyncio.run(main())
