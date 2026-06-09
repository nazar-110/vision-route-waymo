"""Evaluation CLI for VisionRoute."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.lidar import lidar_config_from_dict
from src.data.waymo_utils import discover_tfrecords
from src.data.waymo_perception_dataset import WaymoPerceptionMultiSegmentDataset, WaymoPerceptionSequenceDataset
from src.models.baselines import constant_curvature_baseline, constant_velocity_baseline
from src.models.planner import build_model_from_config
from src.training.metrics import metric_dict
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.visualization.bev_plot import save_bev_comparison
from src.visualization.overlay_route import save_route_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisionRoute and baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _normalize_indices(indices: list[int], file_count: int) -> list[int]:
    normalized: list[int] = []
    for index in indices:
        value = file_count + index if index < 0 else index
        if value < 0 or value >= file_count:
            raise IndexError(f"segment index {index} out of range for {file_count} files")
        normalized.append(value)
    return normalized


def build_eval_dataset(cfg: dict, data_dir: str | None):
    data_cfg = cfg.get("data", {})
    mode = str(cfg.get("mode", "")).lower()
    if mode == "waymo_perception_multisegment":
        root = data_dir or str(data_cfg.get("raw_dir", "data/raw/waymo_perception"))
        files = discover_tfrecords(root)
        if len(files) < 2:
            raise RuntimeError(f"Multi-segment evaluation needs at least 2 Perception TFRecords under {root}")
        return WaymoPerceptionMultiSegmentDataset(
            root,
            camera=str(data_cfg.get("camera", "FRONT")),
            image_width=int(data_cfg.get("image_width", 320)),
            image_height=int(data_cfg.get("image_height", 192)),
            past_steps=int(data_cfg.get("past_steps", 16)),
            future_steps=int(data_cfg.get("future_steps", 20)),
            frame_stride=int(data_cfg.get("frame_stride", 2)),
            max_records_per_segment=int(data_cfg.get("max_records_per_segment", data_cfg.get("max_examples", 80))),
            source_file_indices=_normalize_indices([int(x) for x in data_cfg.get("val_file_indices", [-1])], len(files)),
            lidar_config=lidar_config_from_dict(cfg.get("lidar", {})),
        )
    if mode == "waymo_perception_sequence":
        return WaymoPerceptionSequenceDataset(
            data_dir or str(data_cfg.get("raw_dir", "data/raw/waymo_perception")),
            camera=str(data_cfg.get("camera", "FRONT")),
            image_width=int(data_cfg.get("image_width", 320)),
            image_height=int(data_cfg.get("image_height", 192)),
            past_steps=int(data_cfg.get("past_steps", 16)),
            future_steps=int(data_cfg.get("future_steps", 20)),
            frame_stride=int(data_cfg.get("frame_stride", 2)),
            max_records=int(data_cfg.get("max_examples", 120)),
            lidar_config=lidar_config_from_dict(cfg.get("lidar", {})),
        )
    raise ValueError(f"Unsupported data mode: {cfg.get('mode')}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("evaluate")
    output_dir = ensure_dir(args.output_dir or cfg.get("outputs", {}).get("dir", "outputs"))
    checkpoint_path = Path(args.checkpoint or Path(cfg.get("outputs", {}).get("checkpoint_dir", "outputs/checkpoints")) / "best.pt")
    device = choose_device(args.device or cfg.get("training", {}).get("device", "auto"))
    batch_size = int(args.batch_size or cfg.get("training", {}).get("batch_size", 16))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("training", {}).get("num_workers", 0))
    data_cfg = cfg.get("data", {})
    experiment_cfg = cfg.get("experiment", {})
    future_steps = int(data_cfg.get("future_steps", 20))
    dt = float(data_cfg.get("dt", 0.25))

    dataset = build_eval_dataset(cfg, args.data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = build_model_from_config(cfg).to(device)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run training first.")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    pred_all: list[torch.Tensor] = []
    target_all: list[torch.Tensor] = []
    cv_all: list[torch.Tensor] = []
    cc_all: list[torch.Tensor] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["image"], batch["past"], batch["motion"], batch["command"], batch.get("lidar"))
        cv = constant_velocity_baseline(batch["past"], future_steps, dt)
        cc = constant_curvature_baseline(batch["past"], future_steps, dt)
        pred_all.append(pred.cpu())
        target_all.append(batch["future"].cpu())
        cv_all.append(cv.cpu())
        cc_all.append(cc.cpu())

    pred = torch.cat(pred_all)
    target = torch.cat(target_all)
    cv = torch.cat(cv_all)
    cc = torch.cat(cc_all)
    eval_mode = str(cfg.get("mode") or "waymo_perception")
    metrics = {
        "mode": eval_mode,
        "input_contract": experiment_cfg.get("input_contract", "camera_plus_past_ego_motion"),
        "label_source": experiment_cfg.get("label_source", "future_trajectory_labels"),
        "inference_uses_future_labels": bool(experiment_cfg.get("inference_uses_future_labels", False)),
        "neural": metric_dict(pred, target),
        "constant_velocity": metric_dict(cv, target),
        "constant_curvature": metric_dict(cc, target),
    }
    save_json(metrics, Path(output_dir) / "metrics.json")

    if hasattr(dataset, "get_record"):
        rec = dataset.get_record(0)
        item = dataset[0]
        one = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
        one_pred = model(one["image"], one["past"], one["motion"], one["command"], one.get("lidar")).squeeze(0).cpu().numpy()
        one_metrics = metric_dict(torch.from_numpy(one_pred).unsqueeze(0), torch.from_numpy(rec.future).unsqueeze(0))
        try:
            save_route_overlay(
                rec.image,
                rec.calibration,
                Path(output_dir) / "waymo_overlay.png",
                pred=one_pred,
                gt=rec.future,
                history=rec.past,
                frame_text="waymo sample",
                metric_text=f"ADE {one_metrics['ADE']:.2f}  FDE {one_metrics['FDE']:.2f}",
                thickness=int(cfg.get("visualization", {}).get("line_thickness", 8)),
            )
            save_bev_comparison(Path(output_dir) / "bev_comparison.png", rec.past, one_pred, rec.future)
        except ValueError as exc:
            logger.warning("Skipped Waymo overlay render: %s", exc)

    logger.info("Saved metrics to %s", Path(output_dir) / "metrics.json")


if __name__ == "__main__":
    main()
