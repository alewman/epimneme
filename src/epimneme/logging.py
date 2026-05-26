"""Structured JSON logging for Engram.

Provides a JSON formatter that outputs one-line JSON per log record, suitable
for ingestion by Docker log drivers, Loki, CloudWatch, etc.

Enable with:  LOG_FORMAT=json  (environment variable)

Fields emitted:
  ts, level, logger, msg, module, func, line
  + exc (only when an exception is attached)
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            entry["exc"] = "".join(
                traceback.format_exception(*record.exc_info)
            ).rstrip()
        return json.dumps(entry, default=str, ensure_ascii=False)
