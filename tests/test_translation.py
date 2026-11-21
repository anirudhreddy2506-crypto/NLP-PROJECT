"""
Unit tests for Multilingual Translation Engine & Quality Validation (Hindi & Telugu).
"""

import pytest
from src.translation.multilingual_engine import (
    MultilingualTranslator,
    SupportedLanguage,
)
from src.translation.quality_validator import QualityValidator


def test_translation_identity_english():
    """Verify English passthrough maintains formatting and punctuation."""
    translator = MultilingualTranslator()
    res = translator.translate("A red house on a hill", SupportedLanguage.ENGLISH)
    assert res.target_language == "English"
    assert "A red house on a hill" in res.translated_text
    assert res.validation.is_valid is True


def test_translation_hindi_benchmark():
    """Verify benchmark translation matches produce natural Devanagari sentences."""
    translator = MultilingualTranslator()
    res = translator.translate("a red apple on a white plate", SupportedLanguage.HINDI)
    assert res.target_language == "Hindi"
    assert res.script == "Devanagari"
    assert "सेब" in res.translated_text
    assert "प्लेट" in res.translated_text
    assert res.validation.is_valid is True


def test_translation_telugu_benchmark():
    """Verify benchmark translation matches produce natural Telugu script sentences."""
    translator = MultilingualTranslator()
    res = translator.translate("a red apple on a white plate", SupportedLanguage.TELUGU)
    assert res.target_language == "Telugu"
    assert res.script == "Telugu"
    assert "ఆపిల్" in res.translated_text
    assert "ప్లేట్" in res.translated_text
    assert res.validation.is_valid is True


def test_translation_complex_sentences_preserved():
    """Verify complex multi-object sentences preserve descriptive length and details."""
    translator = MultilingualTranslator()
    caption = "A playful brown puppy chasing a ball on the green grass."

    res_hi = translator.translate(caption, SupportedLanguage.HINDI)
    assert res_hi.script == "Devanagari"
    assert len(res_hi.translated_text.split()) >= 8  # Full sentence, not truncated
    assert res_hi.validation.is_valid is True

    res_te = translator.translate(caption, SupportedLanguage.TELUGU)
    assert res_te.script == "Telugu"
    assert len(res_te.translated_text.split()) >= 7  # Full sentence, not truncated
    assert res_te.validation.is_valid is True


def test_quality_validator_catches_untranslated_and_short():
    """Verify QualityValidator catches empty, overly truncated, or wrong-script translations."""
    validator = QualityValidator()

    # Empty translation
    res_empty = validator.validate("A red apple", "", "Hindi")
    assert res_empty.is_valid is False

    # Untranslated English text in Hindi
    res_untranslated = validator.validate("A young girl running", "A young girl running", "Hindi")
    assert res_untranslated.is_valid is False
    assert any("Devanagari" in s or "untranslated" in s for s in res_untranslated.issues)

    # Valid translation
    res_valid = validator.validate("A red apple on a plate", "सफ़ेद प्लेट पर एक लाल सेब।", "Hindi")
    assert res_valid.is_valid is True
    assert res_valid.quality_score >= 80.0
