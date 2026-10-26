"""
Evaluation pipeline for VisionCaption AI.
Compares baseline and fine-tuned models on held-out test datasets using BLEU, ROUGE, and Latency metrics.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logger
from src.captioning.blip_model import BLIPCaptioner
from src.captioning.generation import generate_caption

logger = setup_logger("visioncaption.evaluation")


def compute_metrics(predictions: List[str], references_list: List[List[str]]) -> Dict[str, float]:
    """
    Computes BLEU-1, BLEU-2, BLEU-3, BLEU-4 and ROUGE-L scores.
    Handles multiple reference captions per image.

    Note on Limitations:
        BLEU measures surface n-gram precision against references; it does not measure visual grounding
        or factual faithfulness directly. Higher BLEU indicates lexical alignment with references.
    """
    smoothing = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []

    for pred, refs in zip(predictions, references_list):
        pred_tokens = pred.lower().split()
        ref_tokens_list = [r.lower().split() for r in refs]

        b1 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        b2 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        b3 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        b4 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        bleu3_scores.append(b3)
        bleu4_scores.append(b4)

        # ROUGE against the best matching reference
        best_r1, best_r2, best_rl = 0.0, 0.0, 0.0
        for ref in refs:
            scores = scorer.score(ref.lower(), pred.lower())
            best_r1 = max(best_r1, scores["rouge1"].fmeasure)
            best_r2 = max(best_r2, scores["rouge2"].fmeasure)
            best_rl = max(best_rl, scores["rougeL"].fmeasure)

        rouge1_scores.append(best_r1)
        rouge2_scores.append(best_r2)
        rougel_scores.append(best_rl)

    n = max(1, len(predictions))
    return {
        "BLEU-1": round(float(sum(bleu1_scores) / n), 4),
        "BLEU-2": round(float(sum(bleu2_scores) / n), 4),
        "BLEU-3": round(float(sum(bleu3_scores) / n), 4),
        "BLEU-4": round(float(sum(bleu4_scores) / n), 4),
        "ROUGE-1": round(float(sum(rouge1_scores) / n), 4),
        "ROUGE-2": round(float(sum(rouge2_scores) / n), 4),
        "ROUGE-L": round(float(sum(rougel_scores) / n), 4),
    }


def evaluate_model(
    model_path: str,
    test_split_path: str = "data/processed/test_split.json",
    num_beams: int = 4,
    device_preference: str = "auto",
) -> Dict[str, Any]:
    """
    Evaluates a single model checkpoint on the held-out test split.
    """
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"Held-out test split not found at: {test_split_path}")

    with open(test_split_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    logger.info("Evaluating model '%s' on %d test records...", model_path, len(test_records))
    captioner = BLIPCaptioner(model_name_or_path=model_path, device_preference=device_preference)

    predictions = []
    references = []
    latencies = []
    detailed_rows = []

    for item in test_records:
        img_ref = item.get("image_path") or item.get("image")
        if isinstance(img_ref, str):
            image = Image.open(img_ref).convert("RGB")
        else:
            image = img_ref.convert("RGB")

        ref_caps = item.get("captions") or item.get("caption")
        if isinstance(ref_caps, str):
            ref_caps = [ref_caps]

        result = generate_caption(captioner, image, num_beams=num_beams)
        pred_caption = result["caption"]

        predictions.append(pred_caption)
        references.append(ref_caps)
        latencies.append(result["latency_ms"])

        detailed_rows.append({
            "image_id": item.get("image_id", "unknown"),
            "model": model_path,
            "prediction": pred_caption,
            "references": " | ".join(ref_caps),
            "latency_ms": result["latency_ms"],
        })

    metrics = compute_metrics(predictions, references)
    avg_latency = round(sum(latencies) / max(1, len(latencies)), 1)
    metrics["avg_latency_ms"] = avg_latency
    metrics["model"] = model_path
    metrics["test_samples"] = len(test_records)

    return {
        "metrics": metrics,
        "detailed_predictions": detailed_rows,
    }


def run_comparison(
    baseline_model: str = "Salesforce/blip-image-captioning-base",
    fine_tuned_model: str = "models/fine_tuned",
    test_split_path: str = "data/processed/test_split.json",
    output_dir: str = "outputs/eval_results",
    num_beams: int = 4,
) -> pd.DataFrame:
    """
    Executes benchmark comparison between baseline and fine-tuned BLIP models.
    Exports results to CSV and JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== Running Baseline Model Evaluation ===")
    baseline_eval = evaluate_model(baseline_model, test_split_path=test_split_path, num_beams=num_beams)

    logger.info("=== Running Fine-Tuned Model Evaluation ===")
    fine_tuned_eval = evaluate_model(fine_tuned_model, test_split_path=test_split_path, num_beams=num_beams)

    df_metrics = pd.DataFrame([
        baseline_eval["metrics"],
        fine_tuned_eval["metrics"],
    ])

    csv_path = os.path.join(output_dir, "eval_comparison.csv")
    json_path = os.path.join(output_dir, "eval_comparison.json")

    df_metrics.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": baseline_eval["metrics"],
            "fine_tuned": fine_tuned_eval["metrics"],
            "comparison_table": df_metrics.to_dict(orient="records"),
        }, f, indent=2)

    # Save all detailed sample predictions
    pred_csv_path = os.path.join(output_dir, "predictions_comparison.csv")
    df_preds = pd.DataFrame(baseline_eval["detailed_predictions"] + fine_tuned_eval["detailed_predictions"])
    df_preds.to_csv(pred_csv_path, index=False)

    logger.info("Evaluation comparison saved to %s and %s.", csv_path, json_path)
    return df_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLIP baseline vs fine-tuned model.")
    parser.add_argument("--baseline", type=str, default="Salesforce/blip-image-captioning-base", help="Baseline model.")
    parser.add_argument("--fine_tuned", type=str, default="models/fine_tuned", help="Fine-tuned model checkpoint.")
    parser.add_argument("--test_data", type=str, default="data/processed/test_split.json", help="Held-out test split.")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results", help="Directory for output metrics.")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam count.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = run_comparison(
        baseline_model=args.baseline,
        fine_tuned_model=args.fine_tuned,
        test_split_path=args.test_data,
        output_dir=args.output_dir,
        num_beams=args.num_beams,
    )
    print("\n=== Model Benchmark Comparison ===")
    print(df.to_string(index=False))
