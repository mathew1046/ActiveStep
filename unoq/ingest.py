"""UDP ingest service: ESP32 -> UNO Q.

Receives JSON telemetry packets from the ESP32, stores samples/events/labels
in SQLite, and fans live data out to subscribers (dashboard WebSocket, tier2,
metronome, bandit, fall FSM).

Packet format from the ESP32 (one UDP datagram, JSON):
{
  "seq": 123, "t_ms": 45678,
  "imu": [[ax,ay,az,gx,gy,gz], ...],   # up to 20 samples @100 Hz
  "fi": 1.23, "pfog": 0.42, "state": "IDLE|CUEING|RECOVERING",
  "event": {"type": "cue_start|cue_stop|label", "modality": 5, "label": "TRUE|FALSE"},
  "recovery_ms": 850
}
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import sqlite3
import struct
import time
from collections import deque
from dataclasses import dataclass, field

from .db import connect

UDP_PORT = 5005


@dataclass
class State:
    """Shared live state, read by other services and the dashboard."""

    latest: dict = field(default_factory=dict)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    pfog_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    packets_rx: int = 0
    packets_lost: int = 0
    _last_seq: int = -1

    def publish(self, msg: dict):
        msg["t_rx"] = time.time()  # wall-clock rx stamp for feed arbitration
        self.latest = msg
        self.packets_rx += 1
        seq = msg.get("seq", -1)
        if self._last_seq >= 0 and seq > self._last_seq + 1:
            self.packets_lost += seq - self._last_seq - 1
        self._last_seq = max(self._last_seq, seq)
        self.pfog_history.append((time.time(), msg.get("pfog", 0.0)))
        for q in self.subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


STATE = State()

# Peer IPs learned from incoming packet source addresses, used to relay
# traffic between the two ESP32 nodes when AP client isolation blocks
# broadcast and node-to-node unicast.
PEER_IPS: dict = {"node1": None, "node2": None}
NODE2_TELEMETRY_PORT = 8888   # Node 2 listens here for Node 1 binary packets
NODE1_COMMAND_PORT = 9999     # Node 1 listens here for vibration commands

_NODE1_STRUCT = struct.Struct("<ff")  # struct_leg_telemetry {accMag, gyroMag}


def relay_leg_telemetry(transport, msg: dict):
    """Forward Node 1's JSON imu packet to Node 2 in its native binary format.

    Node 2 expects struct_leg_telemetry {accMag, gyroMag} on :8888. Recompute
    the magnitudes from the newest imu sample so Node 2's own detection loop
    works unchanged, even when the AP blocks Node 1's broadcast.
    """
    node2 = PEER_IPS.get("node2")
    if not node2 or transport is None:
        return
    imu = msg.get("imu") or []
    if not imu:
        return
    s = imu[-1]
    acc = math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
    gx, gy, gz = (s[3], s[4], s[5]) if len(s) >= 6 else (0.0, 0.0, 0.0)
    gyro = math.sqrt(gx * gx + gy * gy + gz * gz)
    transport.sendto(_NODE1_STRUCT.pack(acc, gyro), (node2, NODE2_TELEMETRY_PORT))


class IngestProtocol(asyncio.DatagramProtocol):
    def __init__(self, db_queue: asyncio.Queue):
        self.db_queue = db_queue
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        try:
            msg = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        # Node 2 (gateway) packets carry acc_mag — record its IP for the relay.
        if "acc_mag" in msg:
            PEER_IPS["node2"] = addr[0]
        # Node 1 announces itself via heartbeat ({"node": "node1"}) and used to
        # send vibration+imu packets; learn its IP either way for the relay.
        if msg.get("node") == "node1" or ("vibration" in msg and "imu" in msg):
            PEER_IPS["node1"] = addr[0]
            relay_leg_telemetry(self.transport, msg)  # no-op without imu rows
        STATE.publish(msg)
        try:
            self.db_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass


async def db_writer(db_queue: asyncio.Queue):
    conn = connect()
    while True:
        msg = await db_queue.get()
        try:
            _write_packet(conn, msg)
        except sqlite3.Error:
            pass


def _write_packet(conn: sqlite3.Connection, msg: dict):
    cur = conn.cursor()
    t = msg.get("t_ms", 0)
    for s in msg.get("imu", []):
        ax, ay, az = s[0], s[1], s[2]
        gx, gy, gz = (s[3], s[4], s[5]) if len(s) >= 6 else (0, 0, 0)
        cur.execute(
            "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)",
            (t, "A", ax, ay, az, gx, gy, gz),
        )
    ev = msg.get("event")
    if ev and ev.get("type") == "cue_start":
        cur.execute(
            "INSERT INTO events (t_start, pfog_peak, fi_peak, modality_mask, model_version) VALUES (?,?,?,?,?)",
            (t, msg.get("pfog", 0), msg.get("fi", 0), ev.get("modality", 0), msg.get("model_version", "base")),
        )
        conn.commit()  # need id promptly for labelling
    elif ev and ev.get("type") == "cue_stop" and msg.get("recovery_ms") is not None:
        cur.execute(
            "UPDATE events SET t_end=?, recovery_ms=? WHERE id=(SELECT MAX(id) FROM events)",
            (t, msg["recovery_ms"]),
        )
    elif ev and ev.get("type") == "label":
        last = cur.execute("SELECT MAX(id) FROM events").fetchone()[0]
        if last:
            cur.execute(
                "INSERT INTO labels (event_id, label, t_ms, source) VALUES (?,?,?,?)",
                (last, ev.get("label", "UNKNOWN"), t, "patient"),
            )
    conn.commit()


async def main():
    loop = asyncio.get_running_loop()
    db_queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: IngestProtocol(db_queue), local_addr=("0.0.0.0", UDP_PORT)
    )
    print(f"[ingest] listening on UDP :{UDP_PORT}")
    asyncio.create_task(db_writer(db_queue))
    await asyncio.Event().wait()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
