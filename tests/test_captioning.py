"""
Unit tests for VisionCaption AI image validation, device resolution, and BLIP captioning pipeline.
"""

import io
import pytest
from PIL import Image
import torch

from src.utils.device import get_optimal_device, get_device_info
from src.captioning.generation import validate_and_load_image, generate_caption
from src.captioning.blip_model import BLIPCaptioner


def create_synthetic_image(format="PNG", mode="RGB", size=(100, 100), color=(255, 0, 0)) -> bytes:
    """Helper to generate in-memory synthetic image bytes."""
    img = Image.new(mode, size, color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def test_device_detection():
    """Verify device resolution respects preferences and handles MPS/CPU."""
    cpu_device = get_optimal_device(preference="cpu")
    assert cpu_device.type == "cpu"

    auto_device = get_optimal_device(preference="auto")
    assert auto_device.type in ("mps", "cpu", "cuda")
    info = get_device_info(auto_device)
    assert "device_type" in info


def test_image_validation_valid():
    """Verify valid PNG and JPEG bytes pass validation and convert to RGB."""
    png_bytes = create_synthetic_image(format="PNG", mode="RGBA", color=(10, 20, 30, 255))
    img, meta = validate_and_load_image(png_bytes)
    assert img.mode == "RGB"
    assert meta["format"] == "PNG"
    assert meta["width"] == 100
    assert meta["height"] == 100


def test_image_validation_unsupported_format():
    """Verify unsupported image format (e.g. BMP/TIFF when disallowed) raises ValueError."""
    bmp_bytes = create_synthetic_image(format="BMP", mode="RGB")
    with pytest.raises(ValueError, match="Unsupported format"):
        validate_and_load_image(bmp_bytes)


def test_image_validation_corrupted_data():
    """Verify corrupt bytes raise an image error."""
    corrupt_bytes = b"this is not a valid image file content"
    with pytest.raises(Exception):
        validate_and_load_image(corrupt_bytes)


def test_image_validation_size_limit():
    """Verify images exceeding byte limit are rejected."""
    png_bytes = create_synthetic_image(format="PNG")
    # Set limit to 10 bytes so it immediately triggers size error
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        validate_and_load_image(png_bytes, max_size_bytes=10)


@pytest.mark.slow
def test_blip_caption_generation():
    """Verify BLIP model initializes and generates a non-empty caption."""
    captioner = BLIPCaptioner(
        model_name_or_path="Salesforce/blip-image-captioning-base",
        device_preference="auto",
    )
    img = Image.new("RGB", (224, 224), color=(70, 130, 180))
    result = generate_caption(
        captioner=captioner,
        image=img,
        max_new_tokens=20,
        num_beams=2,
    )
    assert result["success"] is True
    assert isinstance(result["caption"], str)
    assert len(result["caption"]) > 0
    assert result["latency_seconds"] > 0
    assert result["device"] in ("mps", "cpu", "cuda")
