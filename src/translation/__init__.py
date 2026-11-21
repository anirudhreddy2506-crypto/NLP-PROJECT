"""
Multilingual translation module for VisionCaption AI.
Provides neural translation support across English, Hindi, and Telugu,
along with automated linguistic quality validation.
"""

from .multilingual_engine import MultilingualTranslator, SupportedLanguage, TranslationResult
from .quality_validator import QualityValidator, ValidationResult

__all__ = [
    "MultilingualTranslator",
    "SupportedLanguage",
    "TranslationResult",
    "QualityValidator",
    "ValidationResult",
]
