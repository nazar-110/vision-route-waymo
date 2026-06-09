"""Waymo Perception fallback reader for contiguous camera video clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.calibration import CameraCalibration, calibration_from_waymo
from src.data.lidar import LidarBEVConfig, waymo_frame_to_bev
from src.data.transforms import image_to_tensor, resize_image
from src.data.waymo_utils import WaymoDependencyError, discover_tfrecords


@dataclass(slots=True)
class WaymoPerceptionSequenceRecord:
    """One frame from a contiguous Waymo Perception segment."""

    image: np.ndarray
    lidar_bev: np.ndarray
    past: np.ndarray
    future: np.ndarray
    motion: np.ndarray
    command: int
    calibration: CameraCalibration
    timestamp_micros: int
    context_name: str
    source_file: str


def _load_perception_modules() -> tuple[Any, Any]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise WaymoDependencyError("TensorFlow is required to parse Waymo Perception TFRecords") from exc
    try:
        from waymo_open_dataset import dataset_pb2  # type: ignore
    except Exception as exc:
        raise WaymoDependencyError("waymo_open_dataset is required for Perception TFRecords") from exc
    return tf, dataset_pb2


def _decode_front_image_and_calibration(
    frame: Any,
    dataset_pb2: Any,
    image_width: int,
    image_height: int,
    camera_name: str = "FRONT",
) -> tuple[np.ndarray, CameraCalibration]:
    requested = camera_name.upper()
    requested_id = 1 if requested == "FRONT" else None
    image_proto = None
    for candidate in frame.images:
        name = int(candidate.name)
        enum_name = dataset_pb2.CameraName.Name.Name(name)
        if enum_name.upper() == requested or (requested_id is not None and name == requested_id):
            image_proto = candidate
            break
    if image_proto is None:
        raise ValueError(f"Camera {camera_name} not found in Perception frame")

    bgr = cv2.imdecode(np.frombuffer(image_proto.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV failed to decode Perception camera image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]
    resized = resize_image(rgb, image_width, image_height)

    camera_cal = None
    for cal in frame.context.camera_calibrations:
        if int(cal.name) == int(image_proto.name):
            camera_cal = cal
            break
    if camera_cal is None:
        raise ValueError(f"Calibration for camera {camera_name} not found")
    calibration = calibration_from_waymo(camera_cal, orig_w, orig_h)
    sx = image_width / float(orig_w)
    sy = image_height / float(orig_h)
    calibration.width = int(image_width)
    calibration.height = int(image_height)
    calibration.fx *= sx
    calibration.cx *= sx
    calibration.fy *= sy
    calibration.cy *= sy
    return resized, calibration


def _frame_pose_matrix(frame: Any) -> np.ndarray:
    values = np.asarray(list(frame.pose.transform), dtype=np.float32)
    if values.size != 16:
        raise ValueError("Waymo Perception frame has no 4x4 vehicle pose")
    return values.reshape(4, 4)


def _positions_in_current_vehicle(poses: list[np.ndarray], current_index: int, indices: list[int]) -> np.ndarray:
    current_inv = np.linalg.inv(poses[current_index])
    points = []
    for idx in indices:
        position_global = poses[idx] @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        local = current_inv @ position_global
        points.append([float(local[0]), float(local[1])])
    return np.asarray(points, dtype=np.float32)


class WaymoPerceptionSequenceDataset(Dataset):
    """Contiguous FRONT-camera frames with ego trajectories from frame poses.

    Waymo Perception segments are true video-like sequences. This loader keeps
    frames from one drive together and derives
    past/future ego waypoints from the frame pose stream. Future waypoints are
    labels for offline supervised learning only; ``__getitem__`` returns them so
    the trainer can compute a loss, but planner inference uses only image, past
    motion, and command tensors.
    """

    def __init__(
        self,
        data_dir: str | Path,
        camera: str = "FRONT",
        image_width: int = 320,
        image_height: int = 192,
        past_steps: int = 16,
        future_steps: int = 20,
        frame_stride: int = 2,
        max_records: int | None = 80,
        source_file_index: int = 0,
        lidar_config: LidarBEVConfig | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.camera = camera
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.past_steps = int(past_steps)
        self.future_steps = int(future_steps)
        self.frame_stride = int(frame_stride)
        self.max_records = max_records
        self.lidar_config = lidar_config or LidarBEVConfig()
        files = discover_tfrecords(self.data_dir)
        if not files:
            raise FileNotFoundError(f"No Waymo Perception TFRecords found under {self.data_dir}")
        if source_file_index < 0:
            source_file_index = len(files) + source_file_index
        if source_file_index < 0 or source_file_index >= len(files):
            raise IndexError(f"source_file_index {source_file_index} out of range for {len(files)} files")
        self.source_file = files[source_file_index]
        self.records = self._load_records()
        if not self.records:
            raise RuntimeError(
                "No contiguous Perception records could be built. "
                "Try a smaller stride or verify the segment has enough frames."
            )

    def _load_records(self) -> list[WaymoPerceptionSequenceRecord]:
        tf, dataset_pb2 = _load_perception_modules()
        frames: list[Any] = []
        for serialized in tf.data.TFRecordDataset(str(self.source_file), compression_type=""):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytes(serialized.numpy()))
            frames.append(frame)
        poses = [_frame_pose_matrix(frame) for frame in frames]
        records: list[WaymoPerceptionSequenceRecord] = []
        start = self.past_steps * self.frame_stride
        stop = len(frames) - self.future_steps * self.frame_stride
        for current in range(start, max(start, stop)):
            past_indices = [current - step * self.frame_stride for step in range(self.past_steps - 1, -1, -1)]
            future_indices = [current + step * self.frame_stride for step in range(1, self.future_steps + 1)]
            image, calibration = _decode_front_image_and_calibration(
                frames[current],
                dataset_pb2,
                self.image_width,
                self.image_height,
                self.camera,
            )
            lidar_bev = waymo_frame_to_bev(frames[current], self.lidar_config)
            past = _positions_in_current_vehicle(poses, current, past_indices)
            future = _positions_in_current_vehicle(poses, current, future_indices)
            dt = 0.1 * self.frame_stride
            if len(past) >= 3:
                velocity = (past[-1] - past[-2]) / dt
                prev_velocity = (past[-2] - past[-3]) / dt
                accel = (velocity - prev_velocity) / dt
            else:
                velocity = np.zeros(2, dtype=np.float32)
                accel = np.zeros(2, dtype=np.float32)
            motion = np.array(
                [velocity[0], velocity[1], accel[0], accel[1], np.linalg.norm(velocity), 0.0, past[-1, 1], 0.0],
                dtype=np.float32,
            )
            records.append(
                WaymoPerceptionSequenceRecord(
                    image=image,
                    lidar_bev=lidar_bev,
                    past=past.astype(np.float32),
                    future=future.astype(np.float32),
                    motion=motion,
                    command=0,
                    calibration=calibration,
                    timestamp_micros=int(frames[current].timestamp_micros),
                    context_name=str(frames[current].context.name),
                    source_file=str(self.source_file),
                )
            )
            if self.max_records is not None and len(records) >= self.max_records:
                break
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec = self.records[index]
        return {
            "image": image_to_tensor(rec.image),
            "lidar": torch.from_numpy(rec.lidar_bev),
            "past": torch.from_numpy(rec.past),
            "future": torch.from_numpy(rec.future),
            "motion": torch.from_numpy(rec.motion),
            "command": torch.tensor(rec.command, dtype=torch.long),
        }

    def get_record(self, index: int) -> WaymoPerceptionSequenceRecord:
        return self.records[index]


class WaymoPerceptionMultiSegmentDataset(Dataset):
    """Concatenate records from multiple Waymo Perception segment files.

    This keeps split boundaries at the segment level. Use it to train on a set
    of drives and validate on held-out drives instead of randomly mixing frames
    from the same sequence.
    """

    def __init__(
        self,
        data_dir: str | Path,
        camera: str = "FRONT",
        image_width: int = 320,
        image_height: int = 192,
        past_steps: int = 16,
        future_steps: int = 20,
        frame_stride: int = 2,
        max_records_per_segment: int | None = 80,
        source_file_indices: list[int] | None = None,
        lidar_config: LidarBEVConfig | None = None,
    ) -> None:
        files = discover_tfrecords(data_dir)
        if not files:
            raise FileNotFoundError(f"No Waymo Perception TFRecords found under {data_dir}")
        if source_file_indices is None:
            source_file_indices = list(range(len(files)))
        self.segment_datasets: list[WaymoPerceptionSequenceDataset] = []
        for index in source_file_indices:
            if index < 0:
                index = len(files) + index
            if index < 0 or index >= len(files):
                raise IndexError(f"source_file_index {index} out of range for {len(files)} files")
            self.segment_datasets.append(
                WaymoPerceptionSequenceDataset(
                    data_dir,
                    camera=camera,
                    image_width=image_width,
                    image_height=image_height,
                    past_steps=past_steps,
                    future_steps=future_steps,
                    frame_stride=frame_stride,
                    max_records=max_records_per_segment,
                    source_file_index=index,
                    lidar_config=lidar_config,
                )
            )
        if not self.segment_datasets:
            raise RuntimeError("No Perception segment datasets were built")
        self.records: list[WaymoPerceptionSequenceRecord] = [
            record for segment in self.segment_datasets for record in segment.records
        ]

    @property
    def context_names(self) -> list[str]:
        return sorted({record.context_name for record in self.records})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec = self.records[index]
        return {
            "image": image_to_tensor(rec.image),
            "lidar": torch.from_numpy(rec.lidar_bev),
            "past": torch.from_numpy(rec.past),
            "future": torch.from_numpy(rec.future),
            "motion": torch.from_numpy(rec.motion),
            "command": torch.tensor(rec.command, dtype=torch.long),
        }

    def get_record(self, index: int) -> WaymoPerceptionSequenceRecord:
        return self.records[index]


def extract_front_camera_images(data_dir: str | Path, max_frames: int = 32) -> list[np.ndarray]:
    """Extract FRONT camera RGB images from Waymo Perception TFRecords."""
    files = discover_tfrecords(data_dir)
    if not files:
        raise FileNotFoundError(f"No Waymo Perception TFRecords found under {data_dir}")
    tf, dataset_pb2 = _load_perception_modules()
    frames: list[np.ndarray] = []
    for path in files:
        for serialized in tf.data.TFRecordDataset(str(path), compression_type=""):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytes(serialized.numpy()))
            for image_proto in frame.images:
                if int(image_proto.name) == 1:
                    bgr = cv2.imdecode(np.frombuffer(image_proto.image, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(frames) >= max_frames:
                return frames
    return frames
