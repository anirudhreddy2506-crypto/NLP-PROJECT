"""
Centralized logging configuration for VisionCaption AI.
Provides clean formatting and dual console/file output without shadowing standard library logging.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "visioncaption",
    log_file: Optional[str] = "outputs/logs/visioncaption.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures and returns a logger instance with console and optional file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning("Could not create file log handler at %s: %s", log_file, e)

    return logger
