"""
Logging configuration for CONSTELLATION project.
Sets up structured logging with appropriate formatters and handlers.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from config.settings import LOG_LEVEL, LOG_DIR


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Create a logger with console and file handlers.
    
    Args:
        name: Logger name (usually __name__ from calling module)
        log_file: Optional specific log file name. If None, uses module name.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (detailed format)
    if log_file is None:
        log_file = f"{name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_path = LOG_DIR / log_file
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger


# Create a default logger for the project
default_logger = setup_logger('constellation')