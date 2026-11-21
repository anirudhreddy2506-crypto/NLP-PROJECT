"""
Visual Claim Verifier for VisionCaption AI.
Evaluates whether extracted claims from candidate captions are factually grounded in the image
using BLIP's multimodal cross-attention likelihood and contrastive support scoring.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image

from src.captioning.blip_model import BLIPCaptioner
from .claim_extractor import ExtractedClaim


@dataclass
class VerificationResult:
    claim: str
    category: str
    status: str  # 'VERIFIED', 'HALLUCINATED', 'UNCERTAIN'
    confidence: float
    visual_delta: float
    explanation: str


class VisualVerifier:
    """
    Assesses factual consistency of candidate image captions against the visual input.
    """

    def __init__(
        self,
        captioner: BLIPCaptioner,
        verification_threshold: float = 0.45,
        hallucination_threshold: float = 0.35,
    ):
        self.captioner = captioner
        self.verification_threshold = verification_threshold
        self.hallucination_threshold = hallucination_threshold

    def verify_claims(
        self,
        image: Image.Image,
        claims: List[ExtractedClaim],
    ) -> List[VerificationResult]:
        """
        Verifies a list of candidate claims against an input image.

        Returns:
            List[VerificationResult]: Detailed verification audit for each claim.
        """
        results: List[VerificationResult] = []

        for c in claims:
            eval_dict = self.captioner.score_text_grounding(image, c.text)
            conf = eval_dict["confidence_score"]
            delta = eval_dict["visual_support_delta"]

            if conf >= self.verification_threshold or delta >= -0.10:
                status = "VERIFIED"
                explanation = f"Strong visual grounding (Confidence: {conf * 100:.1f}%, Δ: {delta:+.2f})"
            elif conf < self.hallucination_threshold and delta < -0.30:
                status = "HALLUCINATED"
                explanation = f"Unsupported by visual evidence (Confidence: {conf * 100:.1f}%, Δ: {delta:+.2f})"
            else:
                status = "UNCERTAIN"
                explanation = f"Ambiguous visual support (Confidence: {conf * 100:.1f}%, Δ: {delta:+.2f})"

            results.append(
                VerificationResult(
                    claim=c.text,
                    category=c.category,
                    status=status,
                    confidence=conf,
                    visual_delta=delta,
                    explanation=explanation,
                )
            )

        return results

    def compute_factuality_metrics(
        self, verification_results: List[VerificationResult]
    ) -> Dict[str, Any]:
        """
        Computes aggregate grounding statistics:
        - Total claims
        - Verified count
        - Hallucinated count
        - Faithfulness Score (%)
        - Hallucination Rate (%)
        """
        if not verification_results:
            return {
                "total_claims": 0,
                "verified_claims": 0,
                "hallucinated_claims": 0,
                "faithfulness_score": 100.0,
                "hallucination_rate": 0.0,
                "is_faithful": True,
            }

        total = len(verification_results)
        verified = sum(1 for r in verification_results if r.status == "VERIFIED")
        hallucinated = sum(1 for r in verification_results if r.status == "HALLUCINATED")

        # Average confidence across verified vs total
        faithfulness = round((verified / total) * 100.0, 1)
        hallucination_rate = round((hallucinated / total) * 100.0, 1)

        return {
            "total_claims": total,
            "verified_claims": verified,
            "hallucinated_claims": hallucinated,
            "faithfulness_score": faithfulness,
            "hallucination_rate": hallucination_rate,
            "is_faithful": hallucinated == 0,
        }

