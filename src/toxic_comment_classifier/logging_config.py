"""Centralized logging configuration with Rich console output and rotating file logs."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Console handler uses Rich's formatter; we only set the message part here
# because Rich already prints the timestamp and level in its own columns.
_CONSOLE_FORMAT = "%(name)s | %(message)s"

# File handler keeps a structured, parseable format for log analysis.
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Rotation policy: 5 MB per file, keep 5 backups (~25 MB total max).
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5


def setup_logging(
    level: LogLevel = "INFO",
    log_dir: str | Path = "logs",
    log_filename: str = "training.log",
) -> None:
    """Configure the root logger with Rich console output and a rotating file handler.

    Idempotent: safe to call multiple times. Removes existing handlers before
    re-applying the configuration.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers so repeat calls don't stack output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Console: Rich handler with colored levels and pretty traces.
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
    )
    console_handler.setFormatter(logging.Formatter(fmt=_CONSOLE_FORMAT))
    root.addHandler(console_handler)

    # File: rotating handler, structured format.
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / log_filename,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt=_FILE_FORMAT, datefmt=_FILE_DATEFMT))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
