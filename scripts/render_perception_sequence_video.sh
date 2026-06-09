#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_RUNNER="${PYTHON_BIN:-$HOME/visionroute-py310/bin/python}"
if [ ! -x "$PYTHON_RUNNER" ]; then
  echo "Waymo Python environment not found at $PYTHON_RUNNER"
  echo "Run: bash scripts/setup_wsl_waymo.sh"
  exit 1
fi
"$PYTHON_RUNNER" -m src.visualization.perception_video \
  --config "${WAYMO_SEQUENCE_CONFIG:-configs/default.yaml}" \
  --data_dir data/raw/waymo_perception \
  --output_dir "${WAYMO_SEQUENCE_OUTPUT:-outputs/waymo_multimodal}" \
  --device cpu \
  --max_frames "${WAYMO_SEQUENCE_FRAMES:-80}" \
  --frame_stride "${WAYMO_SEQUENCE_STRIDE:-2}" \
  "$@"
