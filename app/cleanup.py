"""Retention cleanup script."""

from __future__ import annotations

import logging

from .config import load_config
from .storage import StorageManager


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config()
    manager = StorageManager(config.storage)
    removed = manager.purge_old_sessions()
    logging.info("Cleanup complete. Removed %s expired session(s).", removed)


if __name__ == "__main__":
    main()
