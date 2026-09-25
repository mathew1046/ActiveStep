#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source /opt/anaconda/etc/profile.d/conda.sh
conda activate activestep

# Full software simulation on a laptop:
# - unoq.service: ingest + features + fall + metronome + dashboard (one process;
#   STATE is an in-process asyncio pub/sub, so they must share a loop)
# - Python runtime loop with mock IMU replaying Daphnet S01, sending UDP
#   telemetry to the local ingest listener

tmux new-session -d -s activestep-sim \
    "python -m unoq.service" \; \
    new-window -n runtime "ACTIVESTEP_UNOQ_IP=127.0.0.1 python -m activestep.runner --platform mock --subject 1"

echo "Simulation running. Dashboard: http://$(hostname -I | awk '{print $1}'):8000"
