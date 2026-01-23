"""Structured logging helpers for GI Scribe."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path = Path("local_storage") / "logs") -> None:
    """Initialize root logger with console + file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
