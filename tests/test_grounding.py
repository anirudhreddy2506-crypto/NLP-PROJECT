"""
Unit tests for Visual Grounding and Hallucination Reduction subsystem.
"""

import pytest
from src.grounding.claim_extractor import ClaimExtractor
from src.grounding.verifier import VisualVerifier, VerificationResult
from src.grounding.hallucination_reducer import HallucinationReducer


def test_claim_extractor_basic():
    """Verify entity and attribute extraction from descriptive captions."""
    extractor = ClaimExtractor()
    caption = "A fresh red apple on a clean white plate with a green leaf."
    claims = extractor.extract_claims(caption)

    claim_texts = [c.text for c in claims]
    assert any("apple" in t for t in claim_texts)
    assert any("plate" in t for t in claim_texts)
    assert any("leaf" in t for t in claim_texts)

    # Check categories
    categories = [c.category for c in claims]
    assert "attribute_object" in categories or "object" in categories


def test_claim_extractor_complex():
    """Verify handling of complex sentence clauses."""
    extractor = ClaimExtractor()
    caption = "A small cozy red house with green trees nearby under a blue sky."
    claims = extractor.extract_claims(caption)
    claim_texts = [c.text for c in claims]

    assert any("house" in t for t in claim_texts)
    assert any("sky" in t for t in claim_texts)


def test_verifier_metrics_calculation():
    """Verify calculation of aggregate faithfulness and hallucination rate."""
    verifier = VisualVerifier(captioner=None)

    mock_audit = [
        VerificationResult("apple", "object", "VERIFIED", 0.95, 0.6, "Verified"),
        VerificationResult("plate", "object", "VERIFIED", 0.88, 0.4, "Verified"),
        VerificationResult("car", "object", "HALLUCINATED", 0.12, -1.2, "Unsupported"),
        VerificationResult("dog", "object", "HALLUCINATED", 0.08, -1.5, "Unsupported"),
    ]

    metrics = verifier.compute_factuality_metrics(mock_audit)
    assert metrics["total_claims"] == 4
    assert metrics["verified_claims"] == 2
    assert metrics["hallucinated_claims"] == 2
    assert metrics["faithfulness_score"] == 50.0
    assert metrics["hallucination_rate"] == 50.0
    assert metrics["is_faithful"] is False


def test_prune_hallucinated_phrases():
    """Verify syntactic pruning of hallucinated phrases."""
    original = "A red apple sitting on a plate near a blue sports car."
    hallucinated = ["blue sports car"]
    pruned = HallucinationReducer._prune_hallucinated_phrases(original, hallucinated)

    assert "blue sports car" not in pruned
    assert "apple" in pruned

