"""Logging setup. Call `configure_logging` from entry points only."""

import json as jsonlib
import logging
import sys
from datetime import UTC, datetime

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


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Send log records to stderr, as text or as one JSON object per line.

    Safe to call more than once: the previous handler is replaced.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_FLAG, True)
    handler.setFormatter(
        _JsonFormatter()
        if json
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
