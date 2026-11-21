"""
Hallucination Reducer for VisionCaption AI.
Synthesizes verified, hallucination-reduced captions by eliminating or substituting ungrounded tokens,
or re-generating descriptions conditioned strictly on visually validated claims.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image

from src.captioning.blip_model import BLIPCaptioner
from src.captioning.generation import generate_caption
from .claim_extractor import ClaimExtractor, ExtractedClaim
from .verifier import VisualVerifier, VerificationResult


@dataclass
class GroundedCaptionResult:
    original_caption: str
    grounded_caption: str
    hallucination_detected: bool
    faithfulness_score: float
    hallucination_rate: float
    verified_claims: List[str]
    hallucinated_claims: List[str]
    verification_details: List[Dict[str, Any]]
    reduction_applied: bool
    reduction_notes: str


class HallucinationReducer:
    """
    Coordinates claim extraction, visual verification, and hallucination reduction.
    """

    def __init__(self, captioner: BLIPCaptioner):
        self.captioner = captioner
        self.extractor = ClaimExtractor()
        self.verifier = VisualVerifier(captioner=captioner)

    def analyze_and_reduce(
        self,
        image: Image.Image,
        candidate_caption: str,
        regenerate_if_hallucinated: bool = True,
    ) -> GroundedCaptionResult:
        """
        Audits candidate caption for hallucinations and synthesizes a grounded output.
        """
        # 1. Extract visual claims
        claims = self.extractor.extract_claims(candidate_caption)

        # 2. Visually verify each claim
        audit_results = self.verifier.verify_claims(image, claims)
        metrics = self.verifier.compute_factuality_metrics(audit_results)

        verified_texts = [r.claim for r in audit_results if r.status == "VERIFIED"]
        hallucinated_texts = [r.claim for r in audit_results if r.status == "HALLUCINATED"]

        hallucination_detected = len(hallucinated_texts) > 0
        grounded_caption = candidate_caption
        reduction_applied = False
        notes = "All claims verified against visual input."

        # 3. Apply Hallucination Reduction if unsupported claims were identified
        if hallucination_detected:
            reduction_applied = True
            notes = f"Identified unsupported entities: {', '.join(hallucinated_texts)}. Generated grounded replacement."

            if regenerate_if_hallucinated and verified_texts:
                # Conditioning prompt anchored strictly on verified claims
                anchor_prompt = f"a photo of {', '.join(verified_texts)}"
                regen_result = generate_caption(
                    captioner=self.captioner,
                    image=image,
                    text_prompt=anchor_prompt,
                    num_beams=3,
                    max_new_tokens=40,
                )
                if regen_result["success"] and regen_result["caption"]:
                    grounded_caption = regen_result["caption"].strip()
                    if not grounded_caption.endswith("."):
                        grounded_caption += "."
                else:
                    grounded_caption = self._prune_hallucinated_phrases(candidate_caption, hallucinated_texts)
            else:
                grounded_caption = self._prune_hallucinated_phrases(candidate_caption, hallucinated_texts)

        # Clean punctuation and capitalization
        if grounded_caption:
            grounded_caption = grounded_caption[0].upper() + grounded_caption[1:]
            if not grounded_caption.endswith("."):
                grounded_caption += "."

        return GroundedCaptionResult(
            original_caption=candidate_caption,
            grounded_caption=grounded_caption,
            hallucination_detected=hallucination_detected,
            faithfulness_score=metrics["faithfulness_score"],
            hallucination_rate=metrics["hallucination_rate"],
            verified_claims=verified_texts,
            hallucinated_claims=hallucinated_texts,
            verification_details=[
                {
                    "claim": r.claim,
                    "category": r.category,
                    "status": r.status,
                    "confidence": r.confidence,
                    "visual_delta": r.visual_delta,
                    "explanation": r.explanation,
                }
                for r in audit_results
            ],
            reduction_applied=reduction_applied,
            reduction_notes=notes,
        )

    @staticmethod
    def _prune_hallucinated_phrases(caption: str, hallucinated_phrases: List[str]) -> str:
        """
        Syntactically removes unsupported noun chunks and clauses from caption text.
        """
        text = caption
        for phrase in hallucinated_phrases:
            # Pattern matching with optional prepositions and conjunctions
            pattern = re.compile(
                r"(\b(?:with|and|on|under|next to|near)?\s*(?:a|an|the)?\s*"
                + re.escape(phrase)
                + r"\b[,;]?)",
                re.IGNORECASE,
            )
            text = pattern.sub("", text)

        # Clean trailing artifacts
        text = re.sub(r"\s+", " ", text).strip(" ,;.-")
        if not text:
            text = "An image containing verified visual elements."
        return text

