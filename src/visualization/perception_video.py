"""Render a true same-drive Waymo Perception camera overlay video."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.lidar import lidar_config_from_dict
from src.data.waymo_perception_dataset import WaymoPerceptionSequenceDataset
from src.models.planner import build_model_from_config
from src.training.metrics import metric_dict
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.visualization.bev_plot import save_bev_comparison
from src.visualization.overlay_route import save_route_overlay
from src.visualization.project_points import project_vehicle_points
from src.visualization.render_video import render_overlay_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a contiguous Waymo Perception route overlay video")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default="data/raw/waymo_perception")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--prediction_only", action="store_true", help="Hide ground-truth route and ADE/FDE overlay in the rendered video")
    parser.add_argument("--output_name", default=None, help="Optional MP4 filename inside the output directory")
    parser.add_argument("--source_file_index", type=int, default=None, help="Perception segment file index to render")
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def best_visible_route_index(preds: list, calibrations: list) -> int:
    """Pick a preview frame with the most projected prediction points visible."""
    best_idx = 0
    best_visible_count = -1
    for idx, pred in enumerate(preds):
        _, visible, _ = project_vehicle_points(pred, calibrations[idx], clip=False)
        visible_count = int(visible.sum())
        if visible_count > best_visible_count:
            best_idx = idx
            best_visible_count = visible_count
    return best_idx


@torch.no_grad()
def render_waymo_perception_video(
    config_path: str | Path = "configs/default.yaml",
    data_dir: str | Path = "data/raw/waymo_perception",
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_name: str | None = "cpu",
    max_frames: int = 80,
    frame_stride: int = 2,
    fps: int = 10,
    prediction_only: bool = False,
    output_name: str | None = None,
    source_file_index: int | None = None,
) -> dict[str, object]:
    """Render predictions for one contiguous Waymo Perception segment."""
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    viz_cfg = cfg.get("visualization", {})
    output_dir_path = ensure_dir(output_dir or cfg.get("outputs", {}).get("dir", "outputs/waymo_multimodal"))
    checkpoint_path = Path(
        checkpoint or Path(cfg.get("outputs", {}).get("checkpoint_dir", "outputs/waymo_multimodal/checkpoints")) / "best.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = WaymoPerceptionSequenceDataset(
        data_dir,
        camera=str(data_cfg.get("camera", "FRONT")),
        image_width=int(data_cfg.get("image_width", 320)),
        image_height=int(data_cfg.get("image_height", 192)),
        past_steps=int(data_cfg.get("past_steps", 16)),
        future_steps=int(data_cfg.get("future_steps", 20)),
        frame_stride=int(frame_stride or data_cfg.get("frame_stride", 2)),
        max_records=int(max_frames),
        lidar_config=lidar_config_from_dict(cfg.get("lidar", {})),
        source_file_index=int(
            source_file_index
            if source_file_index is not None
            else data_cfg.get("render_source_file_index", data_cfg.get("val_file_indices", [0])[0])
        ),
    )
    device = choose_device(device_name)
    model = build_model_from_config(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    images = []
    calibrations = []
    preds = []
    gts = []
    histories = []
    metric_texts = []
    ade_values = []
    fde_values = []
    context_names = set()
    timestamps = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
        pred = model(batch["image"], batch["past"], batch["motion"], batch["command"], batch.get("lidar")).squeeze(0).cpu()
        record = dataset.get_record(idx)
        metrics = metric_dict(pred.unsqueeze(0), item["future"].unsqueeze(0))
        ade_values.append(metrics["ADE"])
        fde_values.append(metrics["FDE"])
        images.append(record.image)
        calibrations.append(record.calibration)
        preds.append(pred.numpy())
        gts.append(record.future)
        histories.append(record.past)
        metric_texts.append(f"ADE {metrics['ADE']:.2f}  FDE {metrics['FDE']:.2f}")
        context_names.add(record.context_name)
        timestamps.append(record.timestamp_micros)

    preview_idx = best_visible_route_index(preds, calibrations)
    save_route_overlay(
        images[preview_idx],
        calibrations[preview_idx],
        output_dir_path / ("waymo_sequence_prediction_only_first_frame.png" if prediction_only else "waymo_sequence_first_frame.png"),
        pred=preds[preview_idx],
        gt=None if prediction_only else gts[preview_idx],
        history=histories[preview_idx],
        frame_text=f"sequence frame {preview_idx:03d}",
        metric_text=None if prediction_only else metric_texts[preview_idx],
        thickness=int(viz_cfg.get("line_thickness", 8)),
        require_visible=False,
    )
    save_bev_comparison(output_dir_path / "waymo_sequence_bev.png", histories[preview_idx], preds[preview_idx], gts[preview_idx])
    video_name = output_name or ("waymo_sequence_prediction_only.mp4" if prediction_only else "waymo_sequence_overlay.mp4")
    video_path = output_dir_path / video_name
    render_overlay_video(
        images,
        calibrations,
        preds,
        None if prediction_only else gts,
        video_path,
        histories=histories,
        fps=int(fps),
        metric_texts=None if prediction_only else metric_texts,
        thickness=int(viz_cfg.get("line_thickness", 8)),
        require_visible=False,
    )
    metrics_out = {
        "mode": "waymo_perception_sequence",
        "context_names": sorted(context_names),
        "num_frames": len(dataset),
        "preview_frame_index": int(preview_idx),
        "timestamp_start_micros": min(timestamps),
        "timestamp_end_micros": max(timestamps),
        "frame_stride": int(frame_stride),
        "prediction_only_render": bool(prediction_only),
        "source_file": str(dataset.source_file),
        "input_contract": cfg.get("experiment", {}).get("input_contract", "camera_plus_past_ego_motion"),
        "label_source": cfg.get("experiment", {}).get("label_source", "future_trajectory_labels"),
        "inference_uses_future_labels": bool(cfg.get("experiment", {}).get("inference_uses_future_labels", False)),
        "mean_ADE": float(sum(ade_values) / len(ade_values)),
        "mean_FDE": float(sum(fde_values) / len(fde_values)),
    }
    metrics_path = output_dir_path / "sequence_metrics.json"
    save_json(metrics_out, metrics_path)
    logger = get_logger("perception_video")
    logger.info("Rendered %d contiguous frames from %s", len(dataset), sorted(context_names)[0])
    logger.info("Saved %s", video_path)
    return {
        "video_path": str(video_path),
        "preview_path": str(
            output_dir_path / ("waymo_sequence_prediction_only_first_frame.png" if prediction_only else "waymo_sequence_first_frame.png")
        ),
        "bev_path": str(output_dir_path / "waymo_sequence_bev.png"),
        "metrics_path": str(metrics_path),
        "metrics": metrics_out,
    }


def main() -> None:
    args = parse_args()
    render_waymo_perception_video(
        config_path=args.config,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        fps=args.fps,
        prediction_only=args.prediction_only,
        output_name=args.output_name,
        source_file_index=args.source_file_index,
    )


if __name__ == "__main__":
    main()
