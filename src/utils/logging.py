"""Logging utilities."""

from __future__ import annotations

import logging


def get_logger(name: str = "vision_route") -> logging.Logger:
    """Return a console logger with a compact format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
