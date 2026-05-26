"""Physical text log — writes activity events to disk.

Complements the in-memory ActivityBus with a persistent, human-readable
log file.  Uses Python's RotatingFileHandler for automatic rotation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .activity import ActivityEvent


_text_logger: logging.Logger | None = None


def get_text_logger() -> logging.Logger:
    """Get or create the singleton text logger."""
    global _text_logger
    if _text_logger is not None:
        return _text_logger

    log_dir = Path(os.environ.get("EPIMNEME_LOG_DIR", "/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "activity.log"

    _text_logger = logging.getLogger("engram.activity.textlog")
    _text_logger.setLevel(logging.INFO)
    _text_logger.propagate = False

    if not _text_logger.handlers:
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            str(log_path),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        _text_logger.addHandler(handler)

    return _text_logger


def format_activity_line(event: ActivityEvent) -> str:
    """Format an activity event as a simple timestamped log line."""
    parts = [event.ts, f"[{event.type}]"]
    if event.project:
        parts.append(f"project={event.project}")
    if event.agent:
        parts.append(f"agent={event.agent}")
    parts.append(event.summary)
    return " ".join(parts)


async def log_event_to_disk(event: ActivityEvent) -> None:
    """Write a single activity event to the text log."""
    logger = get_text_logger()
    line = format_activity_line(event)
    logger.info(line)
