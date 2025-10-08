"""Logging configuration for MMomentA."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Configure logging for MMomentA.

    Parameters
    ----------
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        Path to log file (if None, log to console only)
    format_string : str, optional
        Custom log format string

    Returns
    -------
    logging.Logger
        Configured root logger

    Examples
    --------
    >>> logger = setup_logging(level="DEBUG", log_file="mmomenta.log")
    >>> logger.info("Starting calculation")
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger("mmomenta")

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger
