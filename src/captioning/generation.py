"""
Generation pipeline for BLIP image captioning.
Handles image validation, preprocessing, execution timing, and structured responses.
"""

import io
import time
import logging
from typing import Dict, Any, Optional, Tuple, Union
from PIL import Image, UnidentifiedImageError

from .blip_model import BLIPCaptioner

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


def validate_and_load_image(
    image_input: Union[bytes, io.BytesIO, str, Image.Image],
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Validates the input image format, size, and integrity.

    Args:
        image_input: Bytes, file path, BytesIO, or PIL Image.
        max_size_bytes: Maximum allowed byte size.

    Returns:
        Tuple[Image.Image, Dict[str, Any]]: Validated PIL RGB Image and image metadata.

    Raises:
        ValueError: If file size exceeds limit or format is unsupported.
        UnidentifiedImageError: If the image data is corrupted.
    """
    metadata: Dict[str, Any] = {}

    if isinstance(image_input, Image.Image):
        img = image_input
        metadata["format"] = getattr(img, "format", "PIL")
        metadata["original_size"] = img.size
    elif isinstance(image_input, (bytes, io.BytesIO)):
        if isinstance(image_input, bytes):
            byte_size = len(image_input)
            stream = io.BytesIO(image_input)
        else:
            stream = image_input
            stream.seek(0, io.SEEK_END)
            byte_size = stream.tell()
            stream.seek(0)

        if byte_size > max_size_bytes:
            raise ValueError(
                f"Image size ({byte_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit ({max_size_bytes / (1024 * 1024):.1f} MB)."
            )

        try:
            img = Image.open(stream)
            img.verify()  # Verify integrity
            # After verify(), re-open to load image data
            stream.seek(0)
            img = Image.open(stream)
        except Exception as e:
            raise UnidentifiedImageError(f"Corrupt or invalid image file: {e}")

        img_format = (img.format or "").upper()
        if img_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{img_format}'. Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}."
            )

        metadata["format"] = img_format
        metadata["byte_size"] = byte_size
        metadata["original_size"] = img.size
    elif isinstance(image_input, str):
        # File path
        try:
            img = Image.open(image_input)
            img.verify()
            img = Image.open(image_input)
        except Exception as e:
            raise UnidentifiedImageError(f"Could not load image from '{image_input}': {e}")

        img_format = (img.format or "").upper()
        if img_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{img_format}'. Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}."
            )
        metadata["format"] = img_format
        metadata["original_size"] = img.size
    else:
        raise TypeError(f"Unsupported input type for image: {type(image_input)}")

    # Convert to RGB mode (handles RGBA, P, Grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    metadata["mode"] = img.mode
    metadata["width"], metadata["height"] = img.size
    return img, metadata


def generate_caption(
    captioner: BLIPCaptioner,
    image: Image.Image,
    text_prompt: Optional[str] = None,
    num_beams: int = 4,
    max_new_tokens: int = 50,
    min_length: int = 5,
    repetition_penalty: float = 1.2,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """
    Executes caption generation with performance timing and error handling.

    Returns:
        Dict[str, Any]:
            - caption: str
            - latency_seconds: float
            - latency_ms: float
            - device: str
            - model: str
            - success: bool
            - error: Optional[str]
    """
    start_time = time.perf_counter()
    try:
        caption = captioner.generate(
            image=image,
            text_prompt=text_prompt,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        latency = time.perf_counter() - start_time
        return {
            "caption": caption,
            "latency_seconds": round(latency, 3),
            "latency_ms": round(latency * 1000, 1),
            "device": str(captioner.device),
            "model": captioner.model_name_or_path,
            "success": True,
            "error": None,
        }
    except Exception as e:
        latency = time.perf_counter() - start_time
        logger.exception("Error during caption generation: %s", e)
        return {
            "caption": "",
            "latency_seconds": round(latency, 3),
            "latency_ms": round(latency * 1000, 1),
            "device": str(captioner.device),
            "model": captioner.model_name_or_path,
            "success": False,
            "error": str(e),
        }
