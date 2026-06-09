#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_RUNNER="${PYTHON_BIN:-$HOME/visionroute-py310/bin/python}"
if [ ! -x "$PYTHON_RUNNER" ]; then
  echo "Waymo Python environment not found at $PYTHON_RUNNER"
  echo "Run: bash scripts/setup_wsl_waymo.sh"
  exit 1
fi

mkdir -p outputs/video_gallery

"$PYTHON_RUNNER" -m src.visualization.perception_video \
  --config configs/default.yaml \
  --data_dir data/raw/waymo_perception \
  --checkpoint outputs/waymo_multimodal/checkpoints/best.pt \
  --output_dir outputs/video_gallery \
  --device "${WAYMO_DEVICE:-cpu}" \
  --max_frames "${WAYMO_VIDEO_FRAMES:-80}" \
  --frame_stride "${WAYMO_FRAME_STRIDE:-2}" \
  --source_file_index 0 \
  --prediction_only \
  --output_name train_segment_prediction_only.mp4

"$PYTHON_RUNNER" -m src.visualization.perception_video \
  --config configs/default.yaml \
  --data_dir data/raw/waymo_perception \
  --checkpoint outputs/waymo_multimodal/checkpoints/best.pt \
  --output_dir outputs/video_gallery \
  --device "${WAYMO_DEVICE:-cpu}" \
  --max_frames "${WAYMO_VIDEO_FRAMES:-80}" \
  --frame_stride "${WAYMO_FRAME_STRIDE:-2}" \
  --source_file_index -1 \
  --prediction_only \
  --output_name heldout_prediction_only.mp4

"$PYTHON_RUNNER" -m src.visualization.perception_video \
  --config configs/default.yaml \
  --data_dir data/raw/waymo_perception \
  --checkpoint outputs/waymo_multimodal/checkpoints/best.pt \
  --output_dir outputs/video_gallery \
  --device "${WAYMO_DEVICE:-cpu}" \
  --max_frames "${WAYMO_VIDEO_FRAMES:-80}" \
  --frame_stride "${WAYMO_FRAME_STRIDE:-2}" \
  --source_file_index -1 \
  --output_name heldout_with_ground_truth_debug.mp4
