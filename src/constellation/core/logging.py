"""Logging setup. Call `configure_logging` from entry points only."""

import json as jsonlib
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path

_HANDLER_FLAG = "_constellation_handler"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return jsonlib.dumps(payload)


def configure_logging(
    level: str = "INFO", json: bool = False, file: Path | None = None
) -> None:
    """Send log records to stderr, as text or as one JSON object per line.

    With `file`, records go to that file instead. The file is reopened when it is
    renamed or replaced, so an external rotation (newsyslog, logrotate) is safe.
    Safe to call more than once: the previous handler is replaced.
    """
    root = logging.getLogger()
    for old in list(root.handlers):
        if getattr(old, _HANDLER_FLAG, False):
            root.removeHandler(old)
            old.close()

    handler: logging.Handler
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.WatchedFileHandler(file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_FLAG, True)
    handler.setFormatter(
        _JsonFormatter()
        if json
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
