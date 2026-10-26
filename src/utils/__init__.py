"""
Utility modules for device detection and centralized logging.
"""

from .device import get_optimal_device, get_device_info
from .logger import setup_logger

__all__ = ["get_optimal_device", "get_device_info", "setup_logger"]
