"""
BLIP Captioner model management and inference class.
Loads Salesforce/blip-image-captioning-base or local fine-tuned checkpoints.
"""

from typing import Optional, Union
import logging
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from src.utils.device import get_optimal_device

logger = logging.getLogger(__name__)


class BLIPCaptioner:
    """
    Wrapper around HuggingFace's BLIP for conditional and unconditional image captioning.
    """

    def __init__(
        self,
        model_name_or_path: str = "Salesforce/blip-image-captioning-base",
        device_preference: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = get_optimal_device(preference=device_preference)
        self.torch_dtype = torch_dtype or (torch.float32 if self.device.type == "cpu" else torch.float32)
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the BlipProcessor and BlipForConditionalGeneration onto the selected device.
        Includes automatic fallback to CPU if MPS memory or kernel initialization fails.
        """
        logger.info("Loading BLIP processor from '%s'...", self.model_name_or_path)
        try:
            self.processor = BlipProcessor.from_pretrained(self.model_name_or_path)
        except Exception as e:
            logger.error("Failed to load BLIP processor: %s", e)
            raise RuntimeError(f"Could not load processor from {self.model_name_or_path}: {e}")

        logger.info("Loading BLIP model weights onto device '%s'...", self.device)
        try:
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch_dtype,
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("BLIP model successfully loaded and ready on %s.", self.device)
        except Exception as e:
            if self.device.type == "mps":
                logger.warning(
                    "Encountered error loading BLIP on MPS (%s). Falling back to CPU.", e
                )
                self.device = torch.device("cpu")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=torch.float32,
                )
                self.model.to(self.device)
                self.model.eval()
                logger.info("BLIP model successfully loaded on fallback device: CPU.")
            else:
                logger.error("Failed to load BLIP model: %s", e)
                raise RuntimeError(f"Could not load BLIP model from {self.model_name_or_path}: {e}")

    def generate(
        self,
        image: Image.Image,
        text_prompt: Optional[str] = None,
        max_new_tokens: int = 50,
        min_length: int = 5,
        num_beams: int = 4,
        repetition_penalty: float = 1.2,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generates a caption for an input PIL Image.

        Args:
            image: PIL.Image instance.
            text_prompt: Optional conditional prefix string (e.g. 'a photo of').
            max_new_tokens: Maximum number of tokens to generate.
            min_length: Minimum tokens in output.
            num_beams: Number of beams for beam search.
            repetition_penalty: Repetition penalty factor.
            do_sample: If True, uses nucleus sampling instead of greedy/beam search.
            temperature: Sampling temperature when do_sample=True.
            top_p: Top-p probability threshold when do_sample=True.

        Returns:
            str: Cleaned generated caption.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor is not initialized.")

        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Prepare inputs
        try:
            if text_prompt and text_prompt.strip():
                inputs = self.processor(images=image, text=text_prompt.strip(), return_tensors="pt")
            else:
                inputs = self.processor(images=image, return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.error("Error preprocessing inputs for BLIP: %s", e)
            raise ValueError(f"Failed to preprocess image/text inputs: {e}")

        # Generation configuration
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "repetition_penalty": repetition_penalty,
        }

        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": max(0.1, temperature),
                "top_p": min(1.0, max(0.1, top_p)),
            })
        else:
            gen_kwargs.update({
                "num_beams": max(1, num_beams),
                "do_sample": False,
            })

        # Run inference with fallback if MPS operation fails
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            if self.device.type == "mps":
                logger.warning(
                    "MPS execution failed during generation (%s). Retrying inference on CPU.", e
                )
                self.device = torch.device("cpu")
                self.model.to(self.device)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **gen_kwargs)
            else:
                raise e

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption
