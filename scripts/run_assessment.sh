#!/usr/bin/env bash
# ActiveStep model assessment pipeline (see plan.md, phases 1-5).
#
# Usage:
#   bash scripts/run_assessment.sh audit              # dataset inventory -> models/dataset_audit.json
#   bash scripts/run_assessment.sh cache              # build window cache (.cache/windows)
#   bash scripts/run_assessment.sh smoke [tag]        # 1-epoch mechanics check on subjects 1-3
#   bash scripts/run_assessment.sh full  [tag]        # full nested LOPO assessment
#   bash scripts/run_assessment.sh report [tag]       # render markdown report from summary.json
#   bash scripts/run_assessment.sh all    [tag]       # audit -> cache -> full -> report
set -euo pipefail
cd "$(dirname "$0")/.."

source /opt/anaconda/etc/profile.d/conda.sh
conda activate activestep

STAGE="${1:-all}"
TAG="${2:-baseline_v1}"
CACHE_DIR=".cache/windows"

case "$STAGE" in
  audit)
    python -m src.data --audit --out models/dataset_audit.json
    ;;
  cache)
    python -m src.data --build-cache --cache-dir "$CACHE_DIR"
    ;;
  smoke)
    python -m src.train_nested --tag "${TAG}_smoke" --epochs 1 --subjects 1,2,3 \
      --max-windows 400 --n-boot 100
    python -m src.report --summary "models/nested/${TAG}_smoke/summary.json" \
      --out "models/nested/${TAG}_smoke/assessment_report.md"
    ;;
  full)
    python -m src.train_nested --tag "$TAG"
    ;;
  report)
    python -m src.report --summary "models/nested/$TAG/summary.json" \
      --out "models/nested/$TAG/assessment_report.md"
    ;;
  all)
    python -m src.data --audit --out models/dataset_audit.json
    python -m src.data --build-cache --cache-dir "$CACHE_DIR"
    python -m src.train_nested --tag "$TAG"
    python -m src.report --summary "models/nested/$TAG/summary.json" \
      --out "models/nested/$TAG/assessment_report.md"
    ;;
  *)
    echo "Unknown stage: $STAGE (use audit|cache|smoke|full|report|all)" >&2
    exit 1
    ;;
esac
