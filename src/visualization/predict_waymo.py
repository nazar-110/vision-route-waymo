"""Run VisionRoute inference on a Waymo Perception segment."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.perception_video import render_waymo_perception_video


RAW_MEDIA_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict and render an ego route overlay for a Waymo Perception TFRecord segment"
    )
    parser.add_argument("--input", required=True, help="Waymo Perception TFRecord file, or a directory of TFRecords")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/waymo_multimodal/checkpoints/best.pt")
    parser.add_argument("--output_dir", default="outputs/predictions")
    parser.add_argument("--output_name", default="prediction_overlay.mp4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--source_file_index", type=int, default=0)
    parser.add_argument("--show_ground_truth", action="store_true", help="Render ground-truth future route for debugging")
    return parser.parse_args()


def _reject_raw_media(input_path: Path) -> None:
    if input_path.suffix.lower() not in RAW_MEDIA_SUFFIXES:
        return
    raise SystemExit(
        "Unsupported input: raw image/video inputs are not supported by this camera+LiDAR checkpoint. "
        "A standalone MP4/JPG/PNG does not contain Waymo LiDAR range images, camera calibration, "
        "or ego pose history, so the model cannot build its required inputs. "
        "Use a Waymo Perception TFRecord segment, or train a separate camera-only model for raw video."
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    _reject_raw_media(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    result = render_waymo_perception_video(
        config_path=args.config,
        data_dir=input_path,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        fps=args.fps,
        prediction_only=not args.show_ground_truth,
        output_name=args.output_name,
        source_file_index=args.source_file_index,
    )
    print(f"Saved video: {result['video_path']}")
    print(f"Saved preview: {result['preview_path']}")
    print(f"Saved BEV: {result['bev_path']}")
    print(f"Saved metrics: {result['metrics_path']}")


if __name__ == "__main__":
    main()
