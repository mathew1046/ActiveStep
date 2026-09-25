"""Single-process UNO Q service bundle.

STATE (unoq.ingest) is an in-process asyncio pub/sub: every service that
consumes the live telemetry stream must share one event loop. Running
`unoq.ingest`, `unoq.features`, `unoq.fall`, `unoq.metronome`, and
`dashboard.main` as separate processes gives each its own empty STATE.

This module is the supported entry point:

    python -m unoq.service

It runs, in one loop:
- UDP ingest on :5005 + SQLite writer
- ESP32 binary-telemetry bridge on :8888 (2-node hardware, no UNO Q)
- Fall escalation FSM
- Adaptive metronome (skipped if no audio device)
- FastAPI dashboard + WebSocket on :8000
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from statistics import median, pstdev
from urllib.request import urlopen

import uvicorn

from activestep.config import DASHBOARD_PORT, DEMO_METRICS, NODE2_HTTP_URL
from dashboard.main import app as dashboard_app

from .esp32_bridge import BRIDGE_PORT, BridgeProtocol, CommandRelay
from .fall import FallFSM
from .ingest import NODE1_COMMAND_PORT, UDP_PORT, IngestProtocol, db_writer, STATE
from .metronome import Metronome, audio_pump


class Node2HTTPBridge:
    def __init__(self, publish, url: str = NODE2_HTTP_URL):
        self.publish = publish
        self.url = url
        self.seq = 0
        self.freeze_started = None
        self.cueing_until = 0.0
        self.cue_active = False
        self.was_connected = False
        self.last_failure = 0.0
        self.motion_samples = deque(maxlen=3)
        self.step_times = deque(maxlen=30)
        self.step_amplitudes = deque(maxlen=30)
        self.last_step = 0.0
        self.baseline_cadence = 0.0
        self.baseline_amplitude = 0.0

    def fetch(self) -> dict:
        with urlopen(f"{self.url}/data", timeout=1.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def gait_metrics(self, now: float, motion: float, walking: bool) -> tuple[float, float, float]:
        self.motion_samples.append((now, motion))
        if len(self.motion_samples) == 3:
            (_, previous), (peak_time, peak), (_, current) = self.motion_samples
            if peak > previous and peak >= current and peak >= 0.25 and peak_time - self.last_step >= 0.3:
                self.step_times.append(peak_time)
                self.step_amplitudes.append(peak)
                self.last_step = peak_time

        recent_times = [t for t in self.step_times if now - t <= 8.0]
        intervals = [b - a for a, b in zip(recent_times, recent_times[1:]) if 0.3 <= b - a <= 2.0]
        cadence = 60.0 / median(intervals) if intervals and now - recent_times[-1] <= 2.0 else 0.0
        irregularity = min(pstdev(intervals) / median(intervals), 1.0) if len(intervals) >= 3 else 0.0

        recent_amplitudes = list(self.step_amplitudes)[-4:]
        amplitude = median(recent_amplitudes) if recent_amplitudes else 0.0
        if walking and cadence > 0.0 and amplitude > 0.0:
            if self.baseline_cadence == 0.0:
                self.baseline_cadence = cadence
                self.baseline_amplitude = amplitude
            else:
                self.baseline_cadence = 0.98 * self.baseline_cadence + 0.02 * cadence
                self.baseline_amplitude = 0.98 * self.baseline_amplitude + 0.02 * amplitude

        festination = 0.0
        if self.baseline_cadence > 0.0 and self.baseline_amplitude > 0.0:
            cadence_ratio = cadence / self.baseline_cadence
            amplitude_ratio = amplitude / self.baseline_amplitude
            festination = min(max(cadence_ratio - amplitude_ratio, 0.0), 1.0)
        return cadence, irregularity, festination

    def demo_metrics(self, now: float, motion: float) -> tuple[float, float, float]:
        activity = min(max(motion, 0.0), 1.0)
        cadence = 92.0 + 26.0 * activity + 4.0 * math.sin(now * 0.55)
        irregularity = min(0.07 + 0.09 * abs(math.sin(now * 0.37)) + 0.04 * activity, 0.28)
        festination = min(max(0.08 + (cadence - 98.0) / 55.0 + max(0.25 - activity, 0.0) * 0.35, 0.02), 0.75)
        return cadence, irregularity, festination

    def convert(self, data: dict, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        motion = float(data.get("adxlMotion", data.get("legMotion", 0.0)))
        walking = int(data.get("gaitState", 0)) == 1
        event = None

        if motion > 0.25:
            self.freeze_started = None
        elif walking:
            if self.freeze_started is None:
                self.freeze_started = now
        elif self.freeze_started is not None:
            elapsed = now - self.freeze_started
            self.freeze_started = None
            if elapsed >= 0.8:
                self.cueing_until = now + 2.0
                self.cue_active = True
                event = {"type": "cue_start", "modality": 1, "cause": "emergency"}

        if self.cue_active and now >= self.cueing_until:
            self.cue_active = False
            event = {"type": "cue_stop", "modality": 1, "cause": "auto_end"}

        if now < self.cueing_until:
            pfog = 1.0
            state = "CUEING"
        elif self.freeze_started is not None:
            pfog = min(now - self.freeze_started, 1.0)
            state = "WALKING"
        else:
            pfog = 0.0
            state = "WALKING" if walking else "IDLE"

        cadence, irregularity, festination = self.gait_metrics(now, motion, walking)
        if DEMO_METRICS:
            cadence, irregularity, festination = self.demo_metrics(now, motion)
        msg = {
            "seq": self.seq,
            "t_ms": int(now * 1000),
            "source": "node2_http",
            "fi": motion,
            "pfog": pfog,
            "state": state,
            "cadence": cadence,
            "asymmetry": irregularity,
            "festination": festination,
            "demo_metrics": DEMO_METRICS,
            "acc_mag": float(data.get("accMag", motion)),
            "node2_connected": True,
            "node2_ip": data.get("wifi", ""),
        }
        self.seq += 1
        if event:
            msg["event"] = event
            if event["type"] == "cue_stop":
                msg["recovery_ms"] = 2000
        return msg

    async def run(self):
        if not self.url:
            return
        while True:
            try:
                data = await asyncio.to_thread(self.fetch)
                self.publish(self.convert(data))
                self.was_connected = True
            except (OSError, ValueError, json.JSONDecodeError):
                now = time.monotonic()
                if self.was_connected or now - self.last_failure >= 1.0:
                    self.publish({"source": "node2_http", "node2_connected": False, "status": "standby"})
                    self.last_failure = now
                self.was_connected = False
            await asyncio.sleep(0.1)


async def supervised(name: str, factory, restart: bool = True):
    """Run a service coroutine; log crashes instead of killing the bundle."""
    while True:
        try:
            await factory()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[service] {name} crashed: {exc!r}")
            if not restart:
                return
            await asyncio.sleep(2.0)


async def main():
    loop = asyncio.get_running_loop()
    db_queue: asyncio.Queue = asyncio.Queue(maxsize=5000)

    def publish(msg: dict):
        STATE.publish(msg)
        try:
            db_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    transport, _ = await loop.create_datagram_endpoint(
        lambda: IngestProtocol(db_queue), local_addr=("0.0.0.0", UDP_PORT)
    )
    print(f"[service] ingest listening on UDP :{UDP_PORT}")

    # 2-ESP32 setup: Node 1 broadcasts binary telemetry to :8888 (mirrored for
    # the dashboard and relayed to Node 2); Node 2's vibration commands come
    # back via :9999 and are relayed to Node 1 when node-to-node traffic is
    # blocked by AP client isolation.
    relay_transports = []
    for port, factory in (
        (BRIDGE_PORT, lambda: BridgeProtocol(publish)),
        (NODE1_COMMAND_PORT, CommandRelay),
    ):
        try:
            tr, _ = await loop.create_datagram_endpoint(
                factory, local_addr=("0.0.0.0", port)
            )
            relay_transports.append(tr)
            print(f"[service] esp32 bridge/relay listening on UDP :{port}")
        except OSError as exc:
            print(f"[service] esp32 bridge port :{port} unavailable: {exc}")

    metro = Metronome()
    server = uvicorn.Server(
        uvicorn.Config(
            dashboard_app,
            host="0.0.0.0",
            port=DASHBOARD_PORT,
            log_level="warning",
        )
    )
    print(f"[service] dashboard on http://0.0.0.0:{DASHBOARD_PORT}")
    if NODE2_HTTP_URL:
        print(f"[service] polling Node 2 at {NODE2_HTTP_URL}/data")

    tasks = [
        asyncio.create_task(supervised("db_writer", lambda: db_writer(db_queue))),
        asyncio.create_task(supervised("fall", lambda: FallFSM().run())),
        asyncio.create_task(supervised("metronome", lambda: metro.run())),
        asyncio.create_task(supervised("audio_pump", lambda: audio_pump(metro), restart=False)),
        asyncio.create_task(server.serve()),
    ]
    if NODE2_HTTP_URL:
        tasks.append(asyncio.create_task(supervised("node2_http", Node2HTTPBridge(publish).run)))
    try:
        await asyncio.gather(*tasks)
    finally:
        transport.close()
        for tr in relay_transports:
            tr.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
