#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

cp systemd/*.service /etc/systemd/system/
systemctl daemon-reload

for svc in activestep-ingest activestep-dashboard activestep-metronome activestep-features activestep-fall; do
    systemctl enable --now "$svc" || true
done
