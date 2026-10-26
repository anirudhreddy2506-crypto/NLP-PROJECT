"""
Device detection and management utilities.
Supports Apple Silicon MPS (Metal Performance Shaders) with automatic CPU fallback.
"""

import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)


def get_optimal_device(preference: str = "auto", allow_mps_fallback: bool = True) -> torch.device:
    """
    Selects the best available compute device according to system hardware and preferences.

    Args:
        preference: "auto", "mps", "cuda", or "cpu".
        allow_mps_fallback: Whether to allow automatic fallback to CPU if MPS encounters an error.

    Returns:
        torch.device: The resolved PyTorch device.
    """
    pref = preference.lower().strip()

    if pref == "cpu":
        logger.info("Device explicitly configured to CPU.")
        return torch.device("cpu")

    if pref in ("auto", "mps"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Test a tiny tensor allocation to confirm MPS is operational
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                logger.info("Apple Silicon MPS (Metal Performance Shaders) is available and initialized.")
                return torch.device("mps")
            except Exception as e:
                logger.warning("MPS is reported available but failed initialization: %s. Falling back to CPU.", e)
                return torch.device("cpu")
        elif pref == "mps":
            if allow_mps_fallback:
                logger.warning("MPS was requested but is not available on this machine. Falling back to CPU.")
                return torch.device("cpu")
            else:
                raise RuntimeError("MPS requested but not available.")

    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")

    logger.info("Defaulting to CPU device.")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """
    Returns diagnostic details about the active PyTorch device.
    """
    info = {
        "device_type": device.type,
        "is_mps": device.type == "mps",
        "is_cuda": device.type == "cuda",
        "is_cpu": device.type == "cpu",
    }
    if device.type == "mps":
        info["mps_built"] = torch.backends.mps.is_built()
        info["mps_available"] = torch.backends.mps.is_available()
    return info
