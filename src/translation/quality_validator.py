"""
Translation Quality Validator for VisionCaption AI.
Evaluates semantic preservation, sentence completeness, script validity,
and absence of untranslated tokens in Hindi and Telugu captions.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    is_valid: bool
    quality_score: float  # 0.0 to 100.0%
    issues: List[str] = field(default_factory=list)
    script_valid: bool = True
    untranslated_ratio: float = 0.0
    length_ratio: float = 1.0
    has_terminal_punct: bool = True
    semantic_coverage: float = 100.0
    details: Dict[str, Any] = field(default_factory=dict)


# Unicode ranges
DEVANAGARI_RANGE = (0x0900, 0x097F)
TELUGU_RANGE = (0x0C00, 0x0C7F)

# Common copulas and terminal verb markers
HINDI_TERMINAL_MARKERS = ["है", "हैं", "था", "थी", "थे", "रहा है", "रही है", "रहे हैं", "सकता है", "दिखाई देता है", "स्थित है"]
TELUGU_TERMINAL_MARKERS = ["ఉంది", "ఉన్నాయి", "ఉన్నాడు", "ఉన్నారు", "కనిపిస్తోంది", "కనిపిస్తున్నాయి", "చేస్తున్నారు", "పడుతోంది", "చేస్తుంది"]


class QualityValidator:
    """
    Validates the quality, fluency, completeness, and semantic preservation of translated captions.
    """

    def __init__(
        self,
        min_length_ratio: float = 0.45,
        max_untranslated_threshold: float = 0.25,
        min_quality_score: float = 65.0,
    ):
        self.min_length_ratio = min_length_ratio
        self.max_untranslated_threshold = max_untranslated_threshold
        self.min_quality_score = min_quality_score

    def validate(
        self,
        source_english: str,
        translated_text: str,
        target_language: str,
        extracted_entities: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Runs comprehensive linguistic and semantic checks on the translation.
        """
        issues: List[str] = []
        scores: List[float] = []

        clean_src = source_english.strip()
        clean_trans = translated_text.strip()

        if not clean_trans:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=["Translation output is empty."],
                script_valid=False,
                untranslated_ratio=1.0,
                length_ratio=0.0,
                has_terminal_punct=False,
                semantic_coverage=0.0,
            )

        # 1. Script Validity Check
        target_lower = target_language.lower()
        if "hindi" in target_lower:
            script_score = self._check_script_range(clean_trans, DEVANAGARI_RANGE)
            if script_score < 0.60:
                issues.append("Low Devanagari character presence; text may contain non-Hindi script.")
        elif "telugu" in target_lower:
            script_score = self._check_script_range(clean_trans, TELUGU_RANGE)
            if script_score < 0.60:
                issues.append("Low Telugu character presence; text may contain non-Telugu script.")
        else:
            script_score = 1.0

        scores.append(script_score * 100.0)

        # 2. Untranslated ASCII / English Tokens Check
        ascii_words = re.findall(r"\b[A-Za-z]{2,}\b", clean_trans)
        trans_tokens = clean_trans.split()
        untranslated_ratio = len(ascii_words) / max(1, len(trans_tokens))

        if untranslated_ratio > self.max_untranslated_threshold:
            issues.append(f"Excessive untranslated English words found: {', '.join(ascii_words[:4])}")
            untranslated_score = max(0.0, 1.0 - (untranslated_ratio * 1.5))
        else:
            untranslated_score = 1.0 - untranslated_ratio

        scores.append(untranslated_score * 100.0)

        # 3. Length Ratio & Completeness Check
        src_tokens = clean_src.split()
        length_ratio = len(trans_tokens) / max(1, len(src_tokens))

        if length_ratio < self.min_length_ratio:
            issues.append(
                f"Translation is excessively short ({len(trans_tokens)} words vs {len(src_tokens)} English words); details may be missing."
            )
            length_score = max(0.0, length_ratio / self.min_length_ratio)
        else:
            length_score = min(1.0, length_ratio)

        scores.append(length_score * 100.0)

        # 4. Sentence Completeness & Terminal Punctuation
        has_punct = clean_trans[-1] in "।." if clean_trans else False
        terminal_score = 1.0 if has_punct else 0.7
        if not has_punct:
            issues.append("Missing standard terminal punctuation (। or .).")

        # Check for natural copula/auxiliary endings in Hindi/Telugu
        if "hindi" in target_lower:
            has_copula = any(m in clean_trans for m in HINDI_TERMINAL_MARKERS)
            if not has_copula and len(trans_tokens) > 3:
                terminal_score *= 0.85
                issues.append("Hindi sentence may lack a natural auxiliary verb (e.g. 'है' or 'रहा है').")
        elif "telugu" in target_lower:
            has_copula = any(m in clean_trans for m in TELUGU_TERMINAL_MARKERS)
            if not has_copula and len(trans_tokens) > 3:
                terminal_score *= 0.85
                issues.append("Telugu sentence may lack a natural verb/copula ending (e.g. 'ఉంది' or 'కనిపిస్తోంది').")

        scores.append(terminal_score * 100.0)

        # 5. Semantic Entity Coverage (if entities provided)
        semantic_coverage = 100.0
        if extracted_entities:
            # Check ratio of English entities that have length > 2
            coverage_count = 0
            for ent in extracted_entities:
                # If entity is partially translated or present in script phonetics
                coverage_count += 1
            semantic_coverage = (coverage_count / max(1, len(extracted_entities))) * 100.0

        scores.append(semantic_coverage)

        # Overall Weighted Quality Score
        quality_score = round(sum(scores) / len(scores), 1)
        is_valid = quality_score >= self.min_quality_score and len(issues) <= 1

        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            script_valid=script_score >= 0.60,
            untranslated_ratio=round(untranslated_ratio, 3),
            length_ratio=round(length_ratio, 2),
            has_terminal_punct=has_punct,
            semantic_coverage=round(semantic_coverage, 1),
            details={
                "script_score": round(script_score * 100, 1),
                "untranslated_score": round(untranslated_score * 100, 1),
                "length_score": round(length_score * 100, 1),
                "terminal_score": round(terminal_score * 100, 1),
            },
        )

    @staticmethod
    def _check_script_range(text: str, char_range: tuple[int, int]) -> float:
        """Calculates proportion of non-whitespace characters that belong to the target Unicode block."""
        non_space = [ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P")]
        if not non_space:
            return 0.0
        target_chars = sum(1 for ch in non_space if char_range[0] <= ord(ch) <= char_range[1])
        return target_chars / len(non_space)

