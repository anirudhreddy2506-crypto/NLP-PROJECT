"""
BLIP Captioner model management, inference, and multimodal scoring class.
Loads Salesforce/blip-image-captioning-base or local fine-tuned checkpoints.
Supports Apple Silicon MPS acceleration, automatic CPU fallback, and visual grounding evaluation.
"""

import os
import gc
from typing import Optional, Union, Dict, Any
import logging
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from src.utils.device import get_optimal_device

logger = logging.getLogger(__name__)


class BLIPCaptioner:
    """
    Wrapper around HuggingFace's BLIP for conditional, unconditional, and visual grounding inference.
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
        Includes local offline cache resolution and automatic fallback to CPU if MPS encounters an error.
        """
        logger.info("Loading BLIP processor from '%s'...", self.model_name_or_path)
        try:
            # Try offline local files first to avoid network timeout delays
            self.processor = BlipProcessor.from_pretrained(
                self.model_name_or_path, local_files_only=True
            )
        except Exception:
            try:
                self.processor = BlipProcessor.from_pretrained(self.model_name_or_path)
            except Exception as e:
                logger.error("Failed to load BLIP processor: %s", e)
                raise RuntimeError(f"Could not load processor from {self.model_name_or_path}: {e}")

        logger.info("Loading BLIP model weights onto device '%s'...", self.device)
        try:
            try:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                )
            except Exception:
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
                try:
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        self.model_name_or_path,
                        torch_dtype=torch.float32,
                        local_files_only=True,
                    )
                except Exception:
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

    def score_text_grounding(
        self,
        image: Image.Image,
        claim_phrase: str,
    ) -> Dict[str, float]:
        """
        Evaluates visual grounding for a candidate phrase/claim by calculating:
        1. Negative log-likelihood (cross-entropy loss) of the phrase conditioned on the target image.
        2. Contrastive Delta (Δ) against a neutral uniform visual baseline to isolate visual evidence
           from unconditional text priors.

        Returns:
            Dict[str, float]:
                - 'image_loss': float (lower means higher model confidence given the image)
                - 'neutral_loss': float (baseline text prior loss)
                - 'visual_support_delta': float (positive indicates grounded visual support)
                - 'confidence_score': float normalized [0.0, 1.0]
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor is not initialized.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Standardized prompt for entity verification
        text_target = f"a photo of {claim_phrase.strip()}"

        try:
            # Inputs with actual image
            inputs_real = self.processor(images=image, text=text_target, return_tensors="pt")
            inputs_real = {k: v.to(self.device) for k, v in inputs_real.items()}

            # Neutral baseline image (solid gray) of matching size
            neutral_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            inputs_neutral = self.processor(images=neutral_img, text=text_target, return_tensors="pt")
            inputs_neutral = {k: v.to(self.device) for k, v in inputs_neutral.items()}

            with torch.no_grad():
                out_real = self.model(**inputs_real, labels=inputs_real["input_ids"])
                out_neutral = self.model(**inputs_neutral, labels=inputs_neutral["input_ids"])

                real_loss = float(out_real.loss.item())
                neutral_loss = float(out_neutral.loss.item())
                delta = neutral_loss - real_loss

                # Sigmoid-normalized confidence: centered around delta=0.0 with slope 1.8
                # Positive delta means real image explains text better than neutral image
                confidence = 1.0 / (1.0 + torch.exp(torch.tensor(-1.8 * delta)).item())

            return {
                "image_loss": round(real_loss, 4),
                "neutral_loss": round(neutral_loss, 4),
                "visual_support_delta": round(delta, 4),
                "confidence_score": round(confidence, 4),
            }
        except Exception as e:
            logger.error("Error evaluating text grounding score for '%s': %s", claim_phrase, e)
            return {
                "image_loss": 5.0,
                "neutral_loss": 2.5,
                "visual_support_delta": -2.5,
                "confidence_score": 0.05,
            }

    def clear_memory(self) -> None:
        """Frees MPS and CUDA memory caches to ensure stability on 16GB unified memory."""
        gc.collect()
        if self.device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        elif self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
