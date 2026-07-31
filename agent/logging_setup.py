"""Structured-ish logging setup shared by the agent and tools."""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.environ.get("DRONE_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, lvl, logging.INFO))
    # python-socketio/engineio are noisy at DEBUG; keep them at WARNING unless asked.
    for noisy in ("engineio", "socketio", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
