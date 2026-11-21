"""
Visual grounding and hallucination reduction module for VisionCaption AI.
Extracts factual claims, verifies them against visual evidence using BLIP cross-attention,
and generates factually reliable, hallucination-reduced captions.
"""

from .claim_extractor import ClaimExtractor, ExtractedClaim
from .verifier import VisualVerifier, VerificationResult
from .hallucination_reducer import HallucinationReducer, GroundedCaptionResult

__all__ = [
    "ClaimExtractor",
    "ExtractedClaim",
    "VisualVerifier",
    "VerificationResult",
    "HallucinationReducer",
    "GroundedCaptionResult",
]

