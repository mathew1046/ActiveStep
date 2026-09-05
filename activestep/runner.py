"""End-to-end runtime loop for the device-side (shank band / Pi / simulator).

This is a single-process Python implementation of the Tier-1 FOG detector.
It can run on a Raspberry Pi, a laptop for simulation, or be ported to the
UNO Q Linux side in a pinch. For the ESP32 the C++ firmware in
firmware/esp32/ is the real implementation; this Python runner is the
software reference that the C++ firmware follows.
"""

from __future__ import annotations

import json
import socket
import time
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf

from activestep.config import (
    AUDIO,
    LASER,
    METRONOME_BPM_MAX,
    METRONOME_BPM_MIN,
    METRONOME_RATIO,
    SAMPLE_RATE,
    UNOQ_IP,
    UNOQ_UDP_PORT,
    VIBRATION,
    WINDOW_SAMPLES,
)
from activestep.hardware import get_backend
from activestep.hardware.base import CueState, HardwareBackend
from src.features import compute_freeze_index
from src.model import StandardScaler, p_fog_combined


class Runtime:
    """Real-time FOG detection + cueing loop."""

    def __init__(
        self,
        backend: HardwareBackend | None = None,
        tflite_path: Path | None = None,
        scaler_path: Path | None = None,
    ):
        self.backend = backend or get_backend()
        self._scaler = StandardScaler().load(scaler_path or Path("models/final/scaler.json"))
        self._interpreter = tf.lite.Interpreter(
            model_path=str(tflite_path or Path("models/final/model_quantized.tflite"))
        )
        self._interpreter.allocate_tensors()
        self._in_d = self._interpreter.get_input_details()[0]
        self._out_d = self._interpreter.get_output_details()[0]

        self._ring: deque[np.ndarray] = deque(maxlen=WINDOW_SAMPLES)
        self._window = np.zeros((WINDOW_SAMPLES, 3), dtype=np.float32)
        self._idx = 0
        self._sample_count = 0
        self._threshold = 0.7
        self._threshold_offset = 0.0
        self._modality = VIBRATION | LASER | AUDIO
        self._cueing = False
        self._cue_start = 0.0
        self._label_window_end = 0.0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def _sample(self):
        sample = self.backend.imu.read()
        self._ring.append(sample[:3].astype(np.float32))
        self._sample_count += 1

    def _inference(self) -> tuple[float, float]:
        if len(self._ring) < WINDOW_SAMPLES:
            return 0.0, 0.0
        win = np.stack(self._ring, axis=0)
        win_n = self._scaler.transform(win)
        self._interpreter.set_tensor(
            self._in_d["index"], win_n[None].astype(self._in_d["dtype"])
        )
        self._interpreter.invoke()
        p_cnn = float(self._interpreter.get_tensor(self._out_d["index"])[0, 0])
        fi = compute_freeze_index(win_n[None, ...], fs=SAMPLE_RATE)[0]
        return p_fog_combined(p_cnn, fi), fi

    def _send(self, fi: float, pfog: float, state: str, event: dict | None = None):
        self._seq += 1
        t = int(time.monotonic() * 1000)
        imu = np.stack(list(self._ring)[-20:], axis=0).tolist() if len(self._ring) >= 20 else []
        msg = {
            "seq": self._seq,
            "t_ms": t,
            "imu": imu,
            "fi": float(fi),
            "pfog": float(pfog),
            "state": state,
        }
        if event:
            msg["event"] = event
        try:
            self._sock.sendto(json.dumps(msg).encode(), (UNOQ_IP, UNOQ_UDP_PORT))
        except OSError:
            pass

    def _check_commands(self):
        try:
            self._sock.setblocking(False)
            data, _ = self._sock.recvfrom(512)
            cmd = json.loads(data.decode())
            if "threshold_offset" in cmd:
                self._threshold_offset = float(cmd["threshold_offset"])
            if "modality_mask" in cmd:
                self._modality = int(cmd["modality_mask"])
            if "tempo_bpm" in cmd:
                pass  # vibrate in sync if needed
        except (BlockingIOError, json.JSONDecodeError):
            pass

    def _handle_labels(self, fi: float, pfog: float, now: float):
        if now > self._label_window_end:
            return
        sw = self.backend.switches.read()
        if sw.get("true"):
            self._send(fi, pfog, "IDLE", {"type": "label", "label": "TRUE"})
            self._label_window_end = 0.0
        if sw.get("false"):
            self._send(fi, pfog, "IDLE", {"type": "label", "label": "FALSE"})
            self._label_window_end = 0.0

    def run(self):
        print("[runtime] starting loop")
        hop = int(SAMPLE_RATE * 0.25)
        next_infer = 0
        self._sock.bind(("0.0.0.0", 5006))
        t0 = time.monotonic()
        while True:
            self._sample()
            self.backend.tick()

            if self._sample_count - next_infer >= hop:
                next_infer = self._sample_count
                pfog, fi = self._inference()
                thr = self._threshold + self._threshold_offset
                now = time.monotonic()
                state = "IDLE"

                if not self._cueing and pfog >= thr:
                    self._cueing = True
                    self._cue_start = now
                    self.backend.cues.set(
                        CueState(
                            vibration=bool(self._modality & VIBRATION),
                            laser=bool(self._modality & LASER),
                            audio=bool(self._modality & AUDIO),
                        )
                    )
                    self.backend.led.set(True)
                    state = "CUEING"
                    self._label_window_end = now + 10.0
                    self._send(fi, pfog, "CUEING", {"type": "cue_start", "modality": self._modality})

                elif self._cueing and pfog < thr - 0.15:
                    recovery = int((now - self._cue_start) * 1000)
                    self._cueing = False
                    self.backend.cues.set(CueState())
                    self.backend.led.set(False)
                    state = "IDLE"
                    self._send(fi, pfog, "IDLE", {"type": "cue_stop", "recovery_ms": recovery})

                elif self._cueing:
                    state = "CUEING"

                if getattr(self, "_verbose", False):
                    print(f"[runtime] pfog={pfog:.3f} fi={fi:.2f} state={state}")

                self._handle_labels(fi, pfog, now)

                if self._sample_count % (SAMPLE_RATE // 5) == 0:
                    self._send(fi, pfog, state)

                self._check_commands()

            # enforce ~100 Hz sampling
            if getattr(self, "_stop", False):
                break
            elapsed = time.monotonic() - t0
            target = self._sample_count / SAMPLE_RATE
            if elapsed < target:
                time.sleep(target - elapsed)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default=None, help="mock | pi | unoq")
    parser.add_argument("--model", default="models/final/model_quantized.tflite")
    parser.add_argument("--scaler", default="models/final/scaler.json")
    parser.add_argument("--subject", type=int, default=1, help="Daphnet subject for replay")
    parser.add_argument("--duration", type=int, default=0, help="Run for N seconds (0=forever)")
    parser.add_argument("--verbose", action="store_true", help="Print every inference")
    args = parser.parse_args()

    backend = get_backend(args.platform)
    rt = Runtime(backend=backend, tflite_path=Path(args.model), scaler_path=Path(args.scaler))
    rt._verbose = args.verbose
    try:
        if args.duration > 0:
            import threading
            stop = threading.Event()

            def stop_after():
                stop.wait(args.duration)
                rt._stop = True  # cooperative stop flag

            threading.Thread(target=stop_after, daemon=True).start()

        rt.run()
    except KeyboardInterrupt:
        print("[runtime] stopped")


if __name__ == "__main__":
    main()
