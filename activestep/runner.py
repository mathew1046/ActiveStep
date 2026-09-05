"""End-to-end runtime loop for the device-side (shank band / Pi / simulator).

This is a single-process Python implementation of the Tier-1 FOG detector.
It can run on a Raspberry Pi, a laptop for simulation, or be ported to the
UNO Q Linux side in a pinch. For the ESP32 the C++ firmware in
firmware/esp32/ is the real implementation; this Python runner is the
software reference that the C++ firmware follows.

Scoring and cue-state logic live in src/detector.py (shared with the offline
benchmark) so the runtime and the assessment pipeline cannot drift apart.
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
    DEFAULT_PFOG_THRESHOLD,
    LASER,
    SAMPLE_RATE,
    UNOQ_IP,
    UNOQ_UDP_PORT,
    VIBRATION,
    WINDOW_SAMPLES,
)
from activestep.hardware import get_backend
from activestep.hardware.base import CueState, HardwareBackend
from src.detector import DetectorConfig, StreamDetector
from src.model import StandardScaler


def _make_predict_fn(interpreter: tf.lite.Interpreter):
    """Wrap a TFLite interpreter as (N, W, 3) -> (N,) probabilities."""
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]

    def fn(X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            interpreter.set_tensor(in_d["index"], X[i : i + 1].astype(in_d["dtype"]))
            interpreter.invoke()
            out[i] = float(np.asarray(interpreter.get_tensor(out_d["index"])).ravel()[0])
        return out

    return fn


class Runtime:
    """Real-time FOG detection + cueing loop."""

    def __init__(
        self,
        backend: HardwareBackend | None = None,
        tflite_path: Path | None = None,
        scaler_path: Path | None = None,
        recipe_path: Path | None = None,
    ):
        self.backend = backend or get_backend()
        self._scaler = StandardScaler().load(scaler_path or Path("models/final/scaler.json"))
        self._interpreter = tf.lite.Interpreter(
            model_path=str(tflite_path or Path("models/final/model_quantized.tflite"))
        )
        self._interpreter.allocate_tensors()
        recipe_file = recipe_path or Path("models/final/recipe.json")
        recipe = json.loads(recipe_file.read_text()) if recipe_file.exists() else {}

        self.detector = StreamDetector(
            DetectorConfig(
                window_samples=WINDOW_SAMPLES,
                hop_samples=int(SAMPLE_RATE * 0.25),
                w_cnn=float(recipe.get("w_cnn", 0.6)),
                w_fi=float(recipe.get("w_fi", 0.4)),
                fi_threshold=float(recipe.get("fi_threshold", 2.0)),
                fi_power_threshold=float(recipe.get("fi_power_threshold", 0.0)),
                threshold=float(recipe.get("threshold", DEFAULT_PFOG_THRESHOLD)),
                hysteresis=float(recipe.get("hysteresis", 0.15)),
                ema_alpha=float(recipe.get("ema_alpha", 1.0)),
                on_consecutive=int(recipe.get("on_consecutive", 1)),
                off_consecutive=int(recipe.get("off_consecutive", 1)),
                min_cue_ms=int(recipe.get("min_cue_ms", 0)),
                refractory_ms=int(recipe.get("refractory_ms", 0)),
            ),
            predict_fn=_make_predict_fn(self._interpreter),
            scaler=self._scaler,
        )

        self._modality = VIBRATION | LASER | AUDIO
        self._label_window_end = 0.0
        self._recent_imu: deque[np.ndarray] = deque(maxlen=20)
        self._last_fi = 0.0
        self._last_score = 0.0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def _send(self, fi: float, pfog: float, state: str, event: dict | None = None):
        self._seq += 1
        t = int(time.monotonic() * 1000)
        imu = [s.tolist() for s in self._recent_imu]
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
                self.detector.set_threshold_offset(float(cmd["threshold_offset"]))
            if "modality_mask" in cmd:
                self._modality = int(cmd["modality_mask"])
            if "tempo_bpm" in cmd:
                pass  # haptic metronome sync: not yet implemented
        except (BlockingIOError, json.JSONDecodeError):
            pass

    def _handle_labels(self, now: float):
        if now > self._label_window_end:
            return
        sw = self.backend.switches.read()
        if sw.get("true") or sw.get("false"):
            label = "TRUE" if sw.get("true") else "FALSE"
            self._send(self._last_fi, self._last_score, "IDLE",
                       {"type": "label", "label": label})
            self._label_window_end = 0.0

    def run(self):
        print("[runtime] starting loop")
        t0 = time.monotonic()
        sample_count = 0
        last_telem = 0.0
        self._sock.bind(("0.0.0.0", 5006))
        while True:
            sample = self.backend.imu.read()
            self._recent_imu.append(np.asarray(sample, dtype=np.float32))
            sample_count += 1
            self.backend.tick()

            now = time.monotonic()
            decision = self.detector.push(int(now * 1000), np.asarray(sample, dtype=np.float32)[:3])

            if decision is not None:
                state = "CUEING" if decision.cueing else "IDLE"
                self._last_fi = decision.fi
                self._last_score = decision.score
                for ev in decision.events:
                    if ev.type == "cue_start":
                        self.backend.cues.set(
                            CueState(
                                vibration=bool(self._modality & VIBRATION),
                                laser=bool(self._modality & LASER),
                                audio=bool(self._modality & AUDIO),
                            )
                        )
                        self.backend.led.set(True)
                        self._label_window_end = now + 10.0
                        self._send(decision.fi, decision.score, "CUEING",
                                   {"type": "cue_start", "modality": self._modality})
                    elif ev.type == "cue_stop":
                        self.backend.cues.set(CueState())
                        self.backend.led.set(False)
                        self._send(decision.fi, decision.score, "IDLE",
                                   {"type": "cue_stop", "recovery_ms": ev.recovery_ms})

                self._handle_labels(now)

                if now - last_telem >= 0.2:
                    last_telem = now
                    self._send(decision.fi, decision.score, state)

                if getattr(self, "_verbose", False):
                    print(f"[runtime] pfog={decision.score:.3f} fi={decision.fi:.2f} state={state}")

                self._check_commands()

            if getattr(self, "_stop", False):
                break
            elapsed = time.monotonic() - t0
            target = sample_count / SAMPLE_RATE
            if elapsed < target:
                time.sleep(target - elapsed)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default=None, help="mock | pi | unoq")
    parser.add_argument("--model", default="models/final/model_quantized.tflite")
    parser.add_argument("--scaler", default="models/final/scaler.json")
    parser.add_argument("--recipe", default="models/final/recipe.json")
    parser.add_argument("--subject", type=int, default=1, help="Daphnet subject for replay")
    parser.add_argument("--duration", type=int, default=0, help="Run for N seconds (0=forever)")
    parser.add_argument("--verbose", action="store_true", help="Print every inference")
    args = parser.parse_args()

    backend = get_backend(args.platform)
    rt = Runtime(
        backend=backend, tflite_path=Path(args.model),
        scaler_path=Path(args.scaler), recipe_path=Path(args.recipe),
    )
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
