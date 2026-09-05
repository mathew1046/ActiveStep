#!/usr/bin/env bash
set -euo pipefail

# One-time setup for the UNO Q Debian side.
# Run as root or with sudo.

echo "=== ActiveStep UNO Q setup ==="

# Update and install system packages
apt-get update
apt-get install -y \
    python3-pip python3-venv \
    network-manager bluez pulseaudio \
    libportaudio2 libatlas-base-dev \
    tmux git

# Python environment
python3 -m venv /opt/activestep
/opt/activestep/bin/pip install --upgrade pip
/opt/activestep/bin/pip install -r /home/mathew/Arrakis/ActiveStep/requirements.txt

# Bluetooth speaker pairing (interactive — must be done once with speaker on)
# bluetoothctl power on
# bluetoothctl agent on
# bluetoothctl scan on
# bluetoothctl pair <MAC>
# bluetoothctl trust <MAC>
# bluetoothctl connect <MAC>

# NetworkManager AP
nmcli connection add type wifi ifname wlan0 con-name activestep-ap \
    autoconnect yes wifi.mode ap wifi.ssid ActiveStep \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk activestep || true
nmcli connection up activestep-ap || true

# mDNS
apt-get install -y avahi-daemon || true
systemctl enable --now avahi-daemon

echo "=== done ==="
echo "Edit requirements.txt if needed, then run: /opt/activestep/bin/python -m dashboard.main"
