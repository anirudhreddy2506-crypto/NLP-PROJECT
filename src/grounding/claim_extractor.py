"""
Claim and Entity Extractor for VisionCaption AI.
Extracts visual claims (objects, attributes, and actions) from candidate captions.
Designed to run robustly in offline/sandboxed environments without requiring external corpora.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExtractedClaim:
    text: str
    category: str  # 'object', 'attribute_object', 'action'
    start_char: int
    end_char: int


# Common non-informative image meta-words
META_WORDS = {
    "photo", "photograph", "image", "picture", "shot", "view", "scene", "close", "up",
    "background", "foreground", "middle", "side", "top", "bottom", "something", "kind",
    "piece", "pair", "group", "lot", "bunch", "set"
}

# Common visual prepositions and determiners
DETERMINERS = {"a", "an", "the", "this", "that", "some", "few", "several", "all"}
PREPOSITIONS = {
    "on", "in", "at", "under", "over", "above", "below", "beside", "near", "with",
    "without", "against", "behind", "in front of", "next to", "along", "by", "from", "to"
}

# Common color and material attributes
ATTRIBUTES = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "grey", "golden", "silver", "dark", "bright", "vibrant", "pale", "wooden",
    "metal", "glass", "plastic", "ceramic", "brick", "stone", "small", "large", "big",
    "tall", "short", "long", "round", "square", "clean", "fresh", "ripe", "cozy", "sunny"
}


class ClaimExtractor:
    """
    Parses natural language image captions into grounded visual claims.
    Combines rule-based dependency chunking and entity heuristics.
    """

    def __init__(self):
        # Compiled patterns for noun chunks and adjective-noun compounds
        self.chunk_splitter = re.compile(r"[,;]|\band\b|\bwith\b|\bwhile\b|\bshowing\b", re.IGNORECASE)

    def extract_claims(self, caption: str) -> List[ExtractedClaim]:
        """
        Extracts atomic visual claims (objects, attribute-object pairs) from caption text.

        Example:
            'a red apple on a white ceramic plate'
            -> [ExtractedClaim('red apple'), ExtractedClaim('white ceramic plate')]
        """
        if not caption or not caption.strip():
            return []

        cleaned = caption.strip().rstrip(".")
        claims: List[ExtractedClaim] = []
        seen_texts = set()

        # Split into syntactic segments
        segments = self.chunk_splitter.split(cleaned)

        for segment in segments:
            tokens = [t.strip().lower() for t in re.findall(r"[A-Za-z0-9\-]+", segment)]
            if not tokens:
                continue

            # Strip leading determiners/prepositions
            filtered_tokens = []
            for t in tokens:
                if not filtered_tokens and (t in DETERMINERS or t in PREPOSITIONS):
                    continue
                filtered_tokens.append(t)

            if not filtered_tokens:
                continue

            # Check for prepositional splits within segment (e.g. 'apple on a plate')
            current_sub: List[str] = []
            for token in filtered_tokens:
                if token in PREPOSITIONS:
                    if current_sub:
                        self._add_claim_if_valid(current_sub, caption, claims, seen_texts)
                        current_sub = []
                elif token in DETERMINERS:
                    if current_sub:
                        self._add_claim_if_valid(current_sub, caption, claims, seen_texts)
                        current_sub = []
                else:
                    current_sub.append(token)

            if current_sub:
                self._add_claim_if_valid(current_sub, caption, claims, seen_texts)

        # Fallback: if no chunks were extracted, extract simple 1-2 word nouns
        if not claims:
            words = [w.lower() for w in re.findall(r"[A-Za-z]+", cleaned)]
            for w in words:
                if w not in DETERMINERS and w not in PREPOSITIONS and w not in META_WORDS and len(w) > 2:
                    match = re.search(r"\b" + re.escape(w) + r"\b", caption, re.IGNORECASE)
                    start = match.start() if match else 0
                    end = match.end() if match else len(w)
                    claims.append(ExtractedClaim(text=w, category="object", start_char=start, end_char=end))

        return claims

    def _add_claim_if_valid(
        self,
        tokens: List[str],
        original_caption: str,
        claims_list: List[ExtractedClaim],
        seen_set: set,
    ) -> None:
        """Helper to validate, classify, and register an extracted claim."""
        # Filter out meta words if standalone
        meaningful = [t for t in tokens if t not in META_WORDS]
        if not meaningful:
            return

        phrase = " ".join(tokens)
        if phrase in seen_set or len(phrase) < 2:
            return

        seen_set.add(phrase)

        # Classify claim category
        has_attr = any(t in ATTRIBUTES for t in tokens)
        category = "attribute_object" if has_attr and len(tokens) > 1 else "object"

        # Locate position in original text
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        match = pattern.search(original_caption)
        start = match.start() if match else -1
        end = match.end() if match else -1

        claims_list.append(
            ExtractedClaim(
                text=phrase,
                category=category,
                start_char=start,
                end_char=end,
            )
        )

