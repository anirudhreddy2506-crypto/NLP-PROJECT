"""
Unit tests for Advanced Evaluation Metrics (METEOR, CIDEr, and Factual Consistency).
"""

import pytest
from src.training.evaluate import (
    compute_meteor_score,
    compute_cider_score,
    compute_metrics,
)


def test_meteor_exact_match():
    """Verify METEOR score for identical candidate and reference."""
    candidate = "a red apple on a white plate"
    references = ["a red apple on a white plate"]
    score = compute_meteor_score(candidate, references)
    assert 0.95 <= score <= 1.0


def test_meteor_partial_match():
    """Verify METEOR score for partially overlapping sentences."""
    candidate = "a red apple sitting on a table"
    references = ["a red apple on a white plate", "an apple on a ceramic table"]
    score = compute_meteor_score(candidate, references)
    assert 0.3 <= score <= 0.9


def test_cider_consensus_scoring():
    """Verify CIDEr score calculation across candidate and multiple references."""
    predictions = [
        "a red house under a blue sky",
        "a fresh apple on a plate",
    ]
    references_list = [
        ["a red house with a green roof under a blue sky", "a red cottage situated on a hill"],
        ["a fresh red apple sitting on a ceramic plate", "a ripe red apple on a clean table"],
    ]
    cider = compute_cider_score(predictions, references_list)
    assert cider > 0.0


def test_compute_metrics_includes_new_metrics():
    """Verify compute_metrics returns all required evaluation metrics."""
    predictions = ["a red apple on a plate"]
    references_list = [["a red apple on a white plate", "an apple on a plate"]]
    metrics = compute_metrics(predictions, references_list, factual_consistency_scores=[92.0])

    assert "BLEU-1" in metrics
    assert "BLEU-4" in metrics
    assert "ROUGE-L" in metrics
    assert "METEOR" in metrics
    assert "CIDEr" in metrics
    assert "Factual_Consistency_%" in metrics
    assert metrics["Factual_Consistency_%"] == 92.0

