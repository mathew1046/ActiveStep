#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source /opt/anaconda/etc/profile.d/conda.sh
conda activate activestep

# Full software simulation on a laptop:
# - ingest + features + fall + dashboard
# - Python runtime loop with mock IMU replaying Daphnet S01

tmux new-session -d -s activestep-sim \
    "python -m unoq.ingest" \; \
    new-window -n features "python -m unoq.features" \; \
    new-window -n fall "python -m unoq.fall" \; \
    new-window -n dashboard "python -m dashboard.main" \; \
    new-window -n runtime "python -m activestep.runner --platform mock --subject 1"

echo "Simulation running. Dashboard: http://$(hostname -I | awk '{print $1}'):8000"
