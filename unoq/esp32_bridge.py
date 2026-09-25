"""Bridge for the 2-ESP32 hardware setup (no UNO Q).

Node 1 (leg sensor) broadcasts a raw struct to 255.255.255.255:8888 @ ~20 Hz:
    struct_leg_telemetry { float accMag; float gyroMag; }   # 8 bytes, little-endian

Node 2 (gateway) consumes that stream and does its own cueing; this bridge
mirrors its detection rule (motion < LEG_WALK_THRESHOLD for FREEZE_CONFIRM_MS
after walking -> cue) purely so the dashboard reflects what the hardware is
doing. For authoritative cue events and GPS, apply the Node 2 firmware patch
that posts JSON straight to ingest :5005.

Run standalone (`python -m unoq.esp32_bridge`, forwards to :5005) or in-bundle
via unoq.service (publishes to STATE in-process).
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import time

from .ingest import (
    NODE1_COMMAND_PORT,
    NODE2_TELEMETRY_PORT,
    PEER_IPS,
    STATE,
    UDP_PORT,
)

BRIDGE_PORT = 8888
LEG_WALK_THRESHOLD = 0.40      # must match Node 2 firmware
FREEZE_CONFIRM_MS = 1000       # must match Node 2 firmware
VIBRATION_DURATION_MS = 4000   # must match Node 1 firmware

_PKT = struct.Struct("<ff")    # accMag, gyroMag


class MirrorFSM:
    """Mirror of Node 2's runFOGDetection() for dashboard display."""

    def __init__(self):
        self.walking = False
        self.freeze_start: float | None = None
        self.cueing_until = 0.0
        self.state_was_cueing = False
        self._seq = 0

    def step(self, acc: float, gyro: float) -> dict:
        self._seq += 1
        now = time.monotonic()
        motion = acc + gyro
        msg = {
            "seq": self._seq,
            "t_ms": int(now * 1000),
            "source": "esp32_bridge",
            "fi": round(motion, 4),
            "pfog": 0.0,
            "state": "IDLE",
        }

        if now < self.cueing_until:
            # Vibration window on Node 1; hold cueing display (event sent once).
            msg["pfog"] = 1.0
            msg["state"] = "CUEING"
            return msg
        if self.state_was_cueing:
            self.state_was_cueing = False
            self.cueing_until = 0.0
            msg["event"] = {"type": "cue_stop", "modality": 1}
            msg["recovery_ms"] = int(VIBRATION_DURATION_MS)

        if motion > LEG_WALK_THRESHOLD:
            self.walking = True
            self.freeze_start = None
            msg["state"] = "WALKING"
        elif self.walking or self.freeze_start is not None:
            if self.freeze_start is None:
                self.freeze_start = now
            elapsed_ms = (now - self.freeze_start) * 1000
            # pFOG ramps 0 -> 1 across the freeze-confirm window.
            msg["pfog"] = min(elapsed_ms / FREEZE_CONFIRM_MS, 1.0)
            msg["state"] = "WALKING" if self.walking else "IDLE"
            if elapsed_ms >= FREEZE_CONFIRM_MS:
                msg["state"] = "CUEING"
                msg["event"] = {"type": "cue_start", "modality": 1}
                self.cueing_until = now + VIBRATION_DURATION_MS / 1000
                self.state_was_cueing = True
                self.walking = False
                self.freeze_start = None
        return msg


class BridgeProtocol(asyncio.DatagramProtocol):
    """Node 1 binary telemetry on :8888 -> mirror FSM + relay to Node 2."""

    def __init__(self, publish):
        self.fsm = MirrorFSM()
        self._publish = publish
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        if len(data) < _PKT.size:
            return
        PEER_IPS["node1"] = addr[0]
        # Relay the raw struct to Node 2 (AP client isolation can block the
        # broadcast). Skip packets that already came from Node 2 — loop guard.
        node2 = PEER_IPS.get("node2")
        if node2 and node2 != addr[0] and self.transport is not None:
            self.transport.sendto(data, (node2, NODE2_TELEMETRY_PORT))

        # Stand down while an authoritative pfog/state feed (:5005 JSON) is live.
        latest = STATE.latest
        if (
            latest.get("source") != "esp32_bridge"
            and "pfog" in latest
            and time.time() - latest.get("t_rx", 0.0) < 2.0
        ):
            return
        acc, gyro = _PKT.unpack_from(data)
        self._publish(self.fsm.step(acc, gyro))


class CommandRelay(asyncio.DatagramProtocol):
    """Vibration commands on :9999 -> Node 1.

    When the relay forwards Node 1 telemetry, Node 2 learns the laptop's IP as
    "node1IP" and posts its commands here; we pass them through to the real
    Node 1. No firmware change needed on either node.
    """

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        node1 = PEER_IPS.get("node1")
        if node1 and node1 != addr[0]:
            self.transport.sendto(data, (node1, NODE1_COMMAND_PORT))


def _publish_to_udp(msg: dict, sock: socket.socket, target: tuple[str, int]):
    sock.sendto(json.dumps(msg).encode(), target)


async def main():
    """Standalone mode: bridge :8888 -> ingest :5005."""
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: BridgeProtocol(lambda m: _publish_to_udp(m, sock, ("127.0.0.1", UDP_PORT))),
        local_addr=("0.0.0.0", BRIDGE_PORT),
    )
    print(f"[bridge] Node 1 telemetry :{BRIDGE_PORT} -> ingest :{UDP_PORT}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
