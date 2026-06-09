#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/predict_waymo.sh <waymo_tfrecord_or_directory>"
  echo "Example: bash scripts/predict_waymo.sh data/raw/waymo_perception/segment-..._with_camera_labels.tfrecord"
  exit 2
fi

PYTHON_RUNNER="${PYTHON_BIN:-$HOME/visionroute-py310/bin/python}"
if [ ! -x "$PYTHON_RUNNER" ]; then
  echo "Waymo Python environment not found at $PYTHON_RUNNER"
  echo "Run: bash scripts/setup_wsl_waymo.sh"
  exit 1
fi

"$PYTHON_RUNNER" -m src.visualization.predict_waymo \
  --input "$1" \
  --config "${VISIONROUTE_CONFIG:-configs/default.yaml}" \
  --checkpoint "${VISIONROUTE_CHECKPOINT:-outputs/waymo_multimodal/checkpoints/best.pt}" \
  --output_dir "${VISIONROUTE_OUTPUT_DIR:-outputs/predictions}" \
  --output_name "${VISIONROUTE_OUTPUT_NAME:-prediction_overlay.mp4}" \
  --device "${WAYMO_DEVICE:-cpu}" \
  --max_frames "${WAYMO_VIDEO_FRAMES:-80}" \
  --frame_stride "${WAYMO_FRAME_STRIDE:-2}" \
  --fps "${WAYMO_FPS:-10}"
