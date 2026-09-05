"""Replay Daphnet data through the UNO Q pipeline.

Used as a demo fallback and for repeatable testing. Reads a subject file,
resamples to 100 Hz, and streams windows at real-time speed to the ingest
service over a local queue.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np

from src.data import load_subject_windows


def packet_stream(windows: np.ndarray, step_s: float = 0.25):
    """Yield telemetry packets from sliding windows."""
    for i, win in enumerate(windows):
        packet = {
            "seq": i,
            "t_ms": int((i * step_s + 2.0) * 1000),
            "imu": win[:, :6].tolist() if win.shape[1] >= 6 else np.concatenate([win, np.zeros((len(win), 3))], axis=1).tolist(),
            "fi": 0.0,
            "pfog": 0.0,
            "state": "IDLE",
        }
        yield packet


async def replay_subject(subject: int, dataset_dir: str, target_ip: str = "127.0.0.1", target_port: int = 5005):
    import socket

    data = load_subject_windows(dataset_dir, subject, window_seconds=2.0, step_seconds=0.25)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
    print(f"[replay] streaming subject {subject:02d}, {len(data['shank'])} windows...")
    for pkt in packet_stream(data["shank"], 0.25):
        expected = pkt["t_ms"] / 1000.0
        sleep = expected - (time.time() - t0)
        if sleep > 0:
            await asyncio.sleep(sleep)
        sock.sendto(json.dumps(pkt).encode(), (target_ip, target_port))
    print("[replay] done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dataset", default="dataset_fog_release/dataset")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    asyncio.run(replay_subject(args.subject, args.dataset, port=args.port))


if __name__ == "__main__":
    main()
