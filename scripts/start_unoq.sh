#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source /opt/anaconda/etc/profile.d/conda.sh
conda activate activestep

echo "Starting ActiveStep UNO Q services..."
# One process hosts ingest + features + fall + metronome + dashboard so they
# share the in-process STATE pub/sub. The ESP32 sends UDP JSON to :5005.
tmux new-session -d -s activestep "python -m unoq.service"

echo "Dashboard at http://$(hostname -I | awk '{print $1}'):8000"
