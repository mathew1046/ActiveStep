"""UNO Q App Lab Linux-side application.

This is the Arduino App Lab Python entry point. It starts the full ActiveStep
service stack (ingest, metronome, features, fall, dashboard) and communicates
with the MCU sketch via the Bridge library.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

# Arduino App Lab exposes the Bridge as an `arduino` module.
try:
    import arduino
    from arduino import Bridge
except ImportError:
    arduino = None
    Bridge = None


def _ensure_wifi_ap():
    """Configure NetworkManager to bring up the UNO Q as an AP."""
    ssid = os.getenv("ACTIVESTEP_SSID", "ActiveStep")
    pw = os.getenv("ACTIVESTEP_PASS", "activestep")
    subprocess.run(
        ["nmcli", "connection", "add", "type", "wifi", "ifname", "wlan0",
         "con-name", "activestep-ap", "autoconnect", "yes",
         "wifi.mode", "ap", "wifi.ssid", ssid, "wifi-sec.key-mgmt", "wpa-psk",
         "wifi-sec.psk", pw],
        check=False,
    )
    subprocess.run(["nmcli", "connection", "up", "activestep-ap"], check=False)


def _start_service(name: str, cmd: list[str]):
    def run():
        print(f"[app] starting {name}")
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=run, daemon=True).start()


def main():
    if arduino is None:
        print("This script is designed for the UNO Q App Lab environment.")
        print("For development run `python -m activestep.runner --platform mock`")
        sys.exit(1)

    _ensure_wifi_ap()

    # Bridge bridge to the MCU sketch. The MCU writes 'trunk_imu' and 'fall_ok',
    # and accepts 'cue_mask' and 'status_led'.
    bridge = Bridge()

    # Start the Linux services. They read the shared state / SQLite.
    _start_service("ingest", [sys.executable, "-m", "unoq.ingest"])
    _start_service("features", [sys.executable, "-m", "unoq.features"])
    _start_service("metronome", [sys.executable, "-m", "unoq.metronome"])
    _start_service("fall", [sys.executable, "-m", "unoq.fall"])
    _start_service("dashboard", [sys.executable, "-m", "dashboard.main"])

    # Forward trunk IMU from Bridge to the local ingest service.
    from unoq.ingest import STATE as live_state

    while True:
        imu_str = bridge.get("trunk_imu")
        if imu_str:
            try:
                vals = [float(x) for x in imu_str.split(",")]
                live_state.latest["trunk_imu"] = vals
            except ValueError:
                pass

        ok = bridge.get("fall_ok")
        if ok:
            live_state.latest["fall_ok"] = True
            bridge.put("fall_ok", "0")  # acknowledge

        # Pull cue/LED commands from the bandit / features services and send to MCU.
        mask = live_state.latest.get("cue_mask", 0)
        bridge.put("cue_mask", str(mask))
        led = live_state.latest.get("status_led", 0)
        bridge.put("status_led", str(int(led)))

        time.sleep(0.01)


if __name__ == "__main__":
    main()
