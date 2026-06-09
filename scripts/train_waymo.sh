#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_RUNNER="${PYTHON_BIN:-$HOME/visionroute-py310/bin/python}"
if [ ! -x "$PYTHON_RUNNER" ]; then
  echo "Waymo Python environment not found at $PYTHON_RUNNER"
  echo "Run: bash scripts/setup_wsl_waymo.sh"
  exit 1
fi

"$PYTHON_RUNNER" -m src.training.train \
  --config configs/default.yaml \
  --data_dir data/raw/waymo_perception \
  --epochs "${WAYMO_EPOCHS:-12}" \
  --batch_size "${WAYMO_BATCH_SIZE:-8}" \
  --device "${WAYMO_DEVICE:-cpu}"

"$PYTHON_RUNNER" -m src.training.evaluate \
  --config configs/default.yaml \
  --data_dir data/raw/waymo_perception \
  --checkpoint outputs/waymo_multimodal/checkpoints/best.pt \
  --batch_size "${WAYMO_BATCH_SIZE:-8}" \
  --device "${WAYMO_DEVICE:-cpu}"
