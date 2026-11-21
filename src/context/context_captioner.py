"""
Context-Aware Captioning Engine for VisionCaption AI.
Implements targeted prompt formulation, hyperparameter adaptation, and style presets:
- Standard / Neutral
- Detailed & Descriptive
- Concise / Short
- Objects-Focused
- Actions & Activities
- Accessibility / Alt-Text (Screen Reader format)
- Custom Instruction
"""

from enum import Enum
import re
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from src.captioning.blip_model import BLIPCaptioner
from src.captioning.generation import generate_caption


class CaptionStyle(str, Enum):
    STANDARD = "Standard / Neutral"
    DETAILED = "Detailed & Descriptive"
    CONCISE = "Concise / Short"
    OBJECTS = "Objects-Focused"
    ACTIONS = "Actions & Activities"
    ACCESSIBILITY = "Accessibility-Focused"
    CUSTOM = "Custom Instruction"


class ContextAwareCaptioner:
    """
    Orchestrates context-aware caption generation using conditional prompts and tuned decoding parameters.
    """

    def __init__(self, captioner: BLIPCaptioner):
        self.captioner = captioner

    @staticmethod
    def get_style_config(
        style: CaptionStyle, custom_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Returns the optimal prompt prefix, token limits, and decoding hyperparameters for a given style.
        """
        if style == CaptionStyle.DETAILED:
            return {
                "prompt_prefix": "a detailed and comprehensive photograph showing",
                "max_new_tokens": 75,
                "min_length": 15,
                "num_beams": 5,
                "repetition_penalty": 1.3,
                "clean_prefix": "a detailed and comprehensive photograph showing",
            }
        elif style == CaptionStyle.CONCISE:
            return {
                "prompt_prefix": "a photo of",
                "max_new_tokens": 20,
                "min_length": 4,
                "num_beams": 3,
                "repetition_penalty": 1.15,
                "clean_prefix": "a photo of",
            }
        elif style == CaptionStyle.OBJECTS:
            return {
                "prompt_prefix": "the main objects and items visible in this image include",
                "max_new_tokens": 50,
                "min_length": 8,
                "num_beams": 4,
                "repetition_penalty": 1.25,
                "clean_prefix": "the main objects and items visible in this image include",
            }
        elif style == CaptionStyle.ACTIONS:
            return {
                "prompt_prefix": "the action, movement, and activity occurring in this image is",
                "max_new_tokens": 55,
                "min_length": 8,
                "num_beams": 4,
                "repetition_penalty": 1.25,
                "clean_prefix": "the action, movement, and activity occurring in this image is",
            }
        elif style == CaptionStyle.ACCESSIBILITY:
            return {
                "prompt_prefix": "alt text for accessibility describing",
                "max_new_tokens": 60,
                "min_length": 10,
                "num_beams": 4,
                "repetition_penalty": 1.2,
                "clean_prefix": "alt text for accessibility describing",
            }
        elif style == CaptionStyle.CUSTOM and custom_instruction and custom_instruction.strip():
            instruction = custom_instruction.strip().rstrip(".:;,")
            return {
                "prompt_prefix": f"a photo showing {instruction}:",
                "max_new_tokens": 60,
                "min_length": 6,
                "num_beams": 4,
                "repetition_penalty": 1.2,
                "clean_prefix": f"a photo showing {instruction}:",
            }
        else:
            # Default / Standard
            return {
                "prompt_prefix": None,
                "max_new_tokens": 50,
                "min_length": 5,
                "num_beams": 4,
                "repetition_penalty": 1.2,
                "clean_prefix": None,
            }

    def generate_context_caption(
        self,
        image: Image.Image,
        style: CaptionStyle = CaptionStyle.STANDARD,
        custom_instruction: Optional[str] = None,
        override_num_beams: Optional[int] = None,
        override_max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Generates a context-conditioned caption according to the requested style preset.
        """
        cfg = self.get_style_config(style, custom_instruction)
        prompt = cfg["prompt_prefix"]
        max_tokens = override_max_new_tokens or cfg["max_new_tokens"]
        beams = override_num_beams or cfg["num_beams"]

        result = generate_caption(
            captioner=self.captioner,
            image=image,
            text_prompt=prompt,
            num_beams=beams,
            max_new_tokens=max_tokens,
            min_length=cfg["min_length"],
            repetition_penalty=cfg["repetition_penalty"],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        if not result["success"]:
            return result

        raw_caption = result["caption"]
        cleaned_caption = self._format_caption(raw_caption, style, cfg.get("clean_prefix"))

        result["caption"] = cleaned_caption
        result["raw_caption"] = raw_caption
        result["style"] = style.value
        result["instruction"] = custom_instruction or ""
        return result

    @staticmethod
    def _format_caption(caption: str, style: CaptionStyle, prefix_to_strip: Optional[str] = None) -> str:
        """
        Cleans and polishes the caption output into natural human-readable prose.
        """
        text = caption.strip()

        # Remove duplicate prefix if the model repeated it verbatim
        if prefix_to_strip:
            pattern = re.compile(f"^{re.escape(prefix_to_strip)}\\s*", re.IGNORECASE)
            text = pattern.sub("", text).strip()

        # Clean trailing colon or punctuation artifact
        text = text.lstrip(": ,.-")
        if not text:
            text = caption.strip()

        # Style-specific formatting wrappers
        if style == CaptionStyle.ACCESSIBILITY:
            if not text.lower().startswith("alt text:"):
                text = f"Alt Text: {text[0].upper()}{text[1:] if len(text) > 1 else ''}"
        elif style == CaptionStyle.OBJECTS:
            if not any(text.lower().startswith(p) for p in ["objects:", "items:", "key objects:"]):
                text = f"Objects present: {text}"
        elif style == CaptionStyle.ACTIONS:
            if not text.lower().startswith("activity:"):
                text = f"Activity: {text}"
        else:
            # Capitalize first letter
            if text:
                text = text[0].upper() + text[1:]

        # Ensure terminal punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text

