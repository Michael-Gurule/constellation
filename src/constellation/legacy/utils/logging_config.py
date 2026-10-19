import logging
import sys

from constellation.legacy.settings import LOG_LEVEL


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Return a console logger. `log_file` is accepted for compatibility and ignored."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    return logger
