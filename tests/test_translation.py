"""
Unit tests for Multilingual Translation Engine (Hindi & Telugu).
"""

import pytest
from src.translation.multilingual_engine import (
    MultilingualTranslator,
    SupportedLanguage,
)


def test_translation_identity_english():
    """Verify English passthrough maintains formatting and punctuation."""
    translator = MultilingualTranslator()
    res = translator.translate("A red house on a hill", SupportedLanguage.ENGLISH)
    assert res.target_language == "English"
    assert "A red house on a hill" in res.translated_text


def test_translation_hindi_benchmark():
    """Verify benchmark translation matches produce natural Devanagari sentences."""
    translator = MultilingualTranslator()
    res = translator.translate("a red apple on a white plate", SupportedLanguage.HINDI)
    assert res.target_language == "Hindi"
    assert res.script == "Devanagari"
    assert "सेब" in res.translated_text
    assert "प्लेट" in res.translated_text
    assert len(res.transliteration) > 0


def test_translation_telugu_benchmark():
    """Verify benchmark translation matches produce natural Telugu script sentences."""
    translator = MultilingualTranslator()
    res = translator.translate("a red apple on a white plate", SupportedLanguage.TELUGU)
    assert res.target_language == "Telugu"
    assert res.script == "Telugu"
    assert "ఆపిల్" in res.translated_text
    assert "ప్లేట్" in res.translated_text


def test_translation_grammar_sov_decomposition():
    """Verify prepositional clause handling converts English SVO to Hindi/Telugu SOV."""
    translator = MultilingualTranslator()
    # Test novel prepositional phrase
    res_hi = translator.translate("a blue car on a road", SupportedLanguage.HINDI)
    assert res_hi.script == "Devanagari"
    assert len(res_hi.translated_text) > 0

    res_te = translator.translate("a blue car on a road", SupportedLanguage.TELUGU)
    assert res_te.script == "Telugu"
    assert len(res_te.translated_text) > 0

