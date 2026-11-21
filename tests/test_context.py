"""
Unit tests for Context-Aware Captioning Engine.
"""

import pytest
from src.context.context_captioner import ContextAwareCaptioner, CaptionStyle


def test_style_configs():
    """Verify prompt prefix and hyperparameters for each style preset."""
    styles = [
        CaptionStyle.STANDARD,
        CaptionStyle.DETAILED,
        CaptionStyle.CONCISE,
        CaptionStyle.OBJECTS,
        CaptionStyle.ACTIONS,
        CaptionStyle.ACCESSIBILITY,
    ]
    for s in styles:
        cfg = ContextAwareCaptioner.get_style_config(s)
        assert "max_new_tokens" in cfg
        assert "num_beams" in cfg
        assert "min_length" in cfg
        assert cfg["max_new_tokens"] >= cfg["min_length"]


def test_custom_instruction_config():
    """Verify custom user instructions are formatted into prompt prefixes."""
    cfg = ContextAwareCaptioner.get_style_config(CaptionStyle.CUSTOM, custom_instruction="describe lighting")
    assert "describe lighting" in cfg["prompt_prefix"]
    assert cfg["max_new_tokens"] > 10


def test_caption_formatting():
    """Verify caption post-processing, capitalization, and prefix stripping."""
    # Accessibility formatting
    raw_acc = "alt text for accessibility describing a red house on a hill"
    cleaned_acc = ContextAwareCaptioner._format_caption(
        raw_acc, CaptionStyle.ACCESSIBILITY, prefix_to_strip="alt text for accessibility describing"
    )
    assert cleaned_acc.startswith("Alt Text:")
    assert cleaned_acc.endswith(".")

    # Objects formatting
    raw_obj = "the main objects and items visible in this image include a red apple and a white plate"
    cleaned_obj = ContextAwareCaptioner._format_caption(
        raw_obj, CaptionStyle.OBJECTS, prefix_to_strip="the main objects and items visible in this image include"
    )
    assert "Objects present:" in cleaned_obj
    assert "red apple" in cleaned_obj

    # Standard capitalization and punctuation
    raw_std = "a red apple on a plate"
    cleaned_std = ContextAwareCaptioner._format_caption(raw_std, CaptionStyle.STANDARD)
    assert cleaned_std == "A red apple on a plate."

