from __future__ import annotations

from src.data.waymo_utils import discover_tfrecords


def test_discover_tfrecords_accepts_direct_file(tmp_path) -> None:
    tfrecord = tmp_path / "segment-example.tfrecord"
    tfrecord.write_bytes(b"")
    assert discover_tfrecords(tfrecord) == [tfrecord.resolve()]


def test_discover_tfrecords_rejects_raw_media_file(tmp_path) -> None:
    raw_video = tmp_path / "drive.mp4"
    raw_video.write_bytes(b"")
    assert discover_tfrecords(raw_video) == []
