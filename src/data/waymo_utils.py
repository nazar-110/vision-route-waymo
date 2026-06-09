"""Shared Waymo data utilities."""

from __future__ import annotations

from pathlib import Path


class WaymoDependencyError(RuntimeError):
    """Raised when Waymo parsing dependencies are missing."""


def discover_tfrecords(data_dir: str | Path) -> list[Path]:
    """Discover TFRecord files without assuming Waymo split counts."""
    root = Path(data_dir)
    if not root.exists():
        return []
    if root.is_file():
        name = root.name.lower()
        if root.suffix.lower() in {".tfrecord", ".tf_record", ".record"} or ".tfrecord" in name:
            return [root.resolve()]
        return []
    patterns = ["*.tfrecord", "*.tf_record", "*.record", "*.tfrecord-*"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})
