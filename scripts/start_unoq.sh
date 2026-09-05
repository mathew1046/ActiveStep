#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source /opt/anaconda/etc/profile.d/conda.sh
conda activate activestep

echo "Starting ActiveStep UNO Q services..."
tmux new-session -d -s activestep "python -m unoq.ingest" \; \
    new-window -n metronome "python -m unoq.metronome" \; \
    new-window -n features "python -m unoq.features" \; \
    new-window -n fall "python -m unoq.fall" \; \
    new-window -n dashboard "python -m dashboard.main"

echo "Dashboard at http://$(hostname -I | awk '{print $1}'):8000"
