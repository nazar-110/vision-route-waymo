"""Training CLI for VisionRoute."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.lidar import lidar_config_from_dict
from src.data.waymo_utils import discover_tfrecords
from src.data.waymo_perception_dataset import WaymoPerceptionMultiSegmentDataset, WaymoPerceptionSequenceDataset
from src.models.losses import combined_loss
from src.models.planner import build_model_from_config
from src.training.metrics import metric_dict
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train camera-conditioned ego trajectory planner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_indices(indices: list[int], file_count: int) -> list[int]:
    normalized: list[int] = []
    for index in indices:
        value = file_count + index if index < 0 else index
        if value < 0 or value >= file_count:
            raise IndexError(f"segment index {index} out of range for {file_count} files")
        normalized.append(value)
    return normalized


def build_datasets(cfg: dict, data_dir: str | None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    data_cfg = cfg.get("data", {})
    seed = int(cfg.get("seed", 7))

    mode = str(cfg.get("mode", "")).lower()
    if mode == "waymo_perception_multisegment":
        root = data_dir or str(data_cfg.get("raw_dir", "data/raw/waymo_perception"))
        files = discover_tfrecords(root)
        if len(files) < 2:
            raise RuntimeError(
                f"Multi-segment training needs at least 2 Perception TFRecords under {root}; found {len(files)}"
            )
        val_indices = _normalize_indices([int(x) for x in data_cfg.get("val_file_indices", [-1])], len(files))
        if "train_file_indices" in data_cfg:
            train_indices = _normalize_indices([int(x) for x in data_cfg["train_file_indices"]], len(files))
        else:
            train_indices = [index for index in range(len(files)) if index not in set(val_indices)]
        common = {
            "camera": str(data_cfg.get("camera", "FRONT")),
            "image_width": int(data_cfg.get("image_width", 320)),
            "image_height": int(data_cfg.get("image_height", 192)),
            "past_steps": int(data_cfg.get("past_steps", 16)),
            "future_steps": int(data_cfg.get("future_steps", 20)),
            "frame_stride": int(data_cfg.get("frame_stride", 2)),
            "max_records_per_segment": int(data_cfg.get("max_records_per_segment", data_cfg.get("max_examples", 80))),
            "lidar_config": lidar_config_from_dict(cfg.get("lidar", {})),
        }
        train = WaymoPerceptionMultiSegmentDataset(root, source_file_indices=train_indices, **common)
        val = WaymoPerceptionMultiSegmentDataset(root, source_file_indices=val_indices, **common)
        return train, val

    if mode == "waymo_perception_sequence":
        root = data_dir or str(data_cfg.get("raw_dir", "data/raw/waymo_perception"))
        full = WaymoPerceptionSequenceDataset(
            root,
            camera=str(data_cfg.get("camera", "FRONT")),
            image_width=int(data_cfg.get("image_width", 320)),
            image_height=int(data_cfg.get("image_height", 192)),
            past_steps=int(data_cfg.get("past_steps", 16)),
            future_steps=int(data_cfg.get("future_steps", 20)),
            frame_stride=int(data_cfg.get("frame_stride", 2)),
            max_records=int(data_cfg.get("max_examples", 120)),
            lidar_config=lidar_config_from_dict(cfg.get("lidar", {})),
        )
        val_size = max(1, int(0.2 * len(full)))
        train_size = len(full) - val_size
        return random_split(full, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    raise ValueError(f"Unsupported data mode: {cfg.get('mode')}")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def run_validation(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["image"], batch["past"], batch["motion"], batch["command"], batch.get("lidar"))
        preds.append(pred.detach().cpu())
        targets.append(batch["future"].detach().cpu())
    return metric_dict(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 7)))
    logger = get_logger("train")

    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    experiment_cfg = cfg.get("experiment", {})
    output_dir = Path(args.output_dir or cfg.get("outputs", {}).get("dir", "outputs"))
    ckpt_dir = ensure_dir(cfg.get("outputs", {}).get("checkpoint_dir", output_dir / "checkpoints"))
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 3))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 16))
    lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 0.001))
    num_workers = int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0))
    device = choose_device(args.device or training_cfg.get("device", "auto"))

    train_ds, val_ds = build_datasets(cfg, data_dir=args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model_from_config(cfg).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    logger.info(
        "Training %s mode on %s: %d train / %d val examples",
        str(cfg.get("mode") or "waymo"),
        device,
        len(train_ds),
        len(val_ds),
    )
    history: list[dict[str, float]] = []
    initial_metrics = run_validation(model, val_loader, device)
    best_ade = initial_metrics["ADE"]
    history.append({"epoch": 0, "train_loss": 0.0, **{f"val_{k}": v for k, v in initial_metrics.items()}})
    initial_checkpoint = {
        "model": model.state_dict(),
        "config": cfg,
        "epoch": 0,
        "metrics": initial_metrics,
        "input_contract": experiment_cfg.get("input_contract", "camera_plus_past_ego_motion"),
        "label_source": experiment_cfg.get("label_source", "future_trajectory_labels"),
        "inference_uses_future_labels": bool(experiment_cfg.get("inference_uses_future_labels", False)),
    }
    torch.save(initial_checkpoint, Path(ckpt_dir) / "best.pt")
    logger.info("epoch 0 train_loss=0.0000 val_ADE=%.3f val_FDE=%.3f", initial_metrics["ADE"], initial_metrics["FDE"])
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["image"], batch["past"], batch["motion"], batch["command"], batch.get("lidar"))
            loss, parts = combined_loss(
                pred,
                batch["future"],
                loss_type=str(training_cfg.get("loss", "smooth_l1")),
                smoothness_weight=float(training_cfg.get("smoothness_weight", 0.02)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.3f}", smooth=f"{parts['smoothness']:.3f}")

        val_metrics = run_validation(model, val_loader, device)
        train_loss = float(np.mean(losses)) if losses else 0.0
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        logger.info("epoch %d train_loss=%.4f val_ADE=%.3f val_FDE=%.3f", epoch, train_loss, val_metrics["ADE"], val_metrics["FDE"])
        checkpoint = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "metrics": val_metrics,
            "input_contract": experiment_cfg.get("input_contract", "camera_plus_past_ego_motion"),
            "label_source": experiment_cfg.get("label_source", "future_trajectory_labels"),
            "inference_uses_future_labels": bool(experiment_cfg.get("inference_uses_future_labels", False)),
        }
        torch.save(checkpoint, Path(ckpt_dir) / "last.pt")
        if val_metrics["ADE"] < best_ade:
            best_ade = val_metrics["ADE"]
            torch.save(checkpoint, Path(ckpt_dir) / "best.pt")

    save_json(
        {
            "history": history,
            "best_val_ADE": best_ade,
            "future_steps": int(data_cfg.get("future_steps", 20)),
            "input_contract": experiment_cfg.get("input_contract", "camera_plus_past_ego_motion"),
            "label_source": experiment_cfg.get("label_source", "future_trajectory_labels"),
            "inference_uses_future_labels": bool(experiment_cfg.get("inference_uses_future_labels", False)),
        },
        output_dir / "train_metrics.json",
    )
    logger.info("Saved best checkpoint to %s", Path(ckpt_dir) / "best.pt")


if __name__ == "__main__":
    main()
