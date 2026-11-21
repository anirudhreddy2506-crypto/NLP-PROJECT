"""
Multilingual Caption Quality Evaluation Suite for VisionCaption AI.
Evaluates English, Hindi, and Telugu caption generations across diverse scene categories:
- People and activities
- Animals
- Nature and landscapes
- Food and objects
- Indoor and outdoor scenes
- Complex multi-object scenes

Measures:
1. Semantic preservation & entity coverage
2. Sentence completeness & word count fidelity
3. Grammatical structure & script validity
4. Naturalness & absence of untranslated tokens
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.translation.multilingual_engine import MultilingualTranslator, SupportedLanguage
from src.translation.quality_validator import QualityValidator
from src.utils.logger import setup_logger

logger = setup_logger("visioncaption.multilingual_eval")


def run_multilingual_benchmark(
    test_suite_path: str = "data/multilingual_test_suite.json",
    output_dir: str = "outputs/eval_results",
) -> pd.DataFrame:
    """
    Executes multilingual evaluation on the test suite across Hindi and Telugu.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(test_suite_path):
        raise FileNotFoundError(f"Test suite not found at {test_suite_path}")

    with open(test_suite_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    translator = MultilingualTranslator()
    validator = QualityValidator()

    results_rows: List[Dict[str, Any]] = []

    logger.info("=== Running Multilingual Evaluation Benchmark (%d categories) ===", len(test_samples))

    for item in test_samples:
        sample_id = item["id"]
        category = item["category"]
        en_cap = item["english_caption"]
        key_elements = item.get("key_elements", [])
        en_word_count = len(en_cap.split())

        # 1. Translate to Hindi
        res_hi = translator.translate(en_cap, SupportedLanguage.HINDI)
        val_hi = validator.validate(en_cap, res_hi.translated_text, "Hindi", key_elements)

        # 2. Translate to Telugu
        res_te = translator.translate(en_cap, SupportedLanguage.TELUGU)
        val_te = validator.validate(en_cap, res_te.translated_text, "Telugu", key_elements)

        hi_word_count = len(res_hi.translated_text.split())
        te_word_count = len(res_te.translated_text.split())

        row = {
            "category": category,
            "english_caption": en_cap,
            "en_words": en_word_count,
            "hindi_caption": res_hi.translated_text,
            "hi_words": hi_word_count,
            "hi_word_ratio": round(hi_word_count / en_word_count, 2),
            "hi_quality_score": val_hi.quality_score,
            "hi_is_valid": val_hi.is_valid,
            "telugu_caption": res_te.translated_text,
            "te_words": te_word_count,
            "te_word_ratio": round(te_word_count / en_word_count, 2),
            "te_quality_score": val_te.quality_score,
            "te_is_valid": val_te.is_valid,
            "method": res_hi.method,
        }
        results_rows.append(row)

    df_eval = pd.DataFrame(results_rows)

    csv_path = os.path.join(output_dir, "multilingual_evaluation.csv")
    json_path = os.path.join(output_dir, "multilingual_evaluation.json")

    df_eval.to_csv(csv_path, index=False)

    summary_stats = {
        "total_test_samples": len(df_eval),
        "avg_english_words": round(float(df_eval["en_words"].mean()), 1),
        "avg_hindi_words": round(float(df_eval["hi_words"].mean()), 1),
        "avg_telugu_words": round(float(df_eval["te_words"].mean()), 1),
        "avg_hindi_quality_score": round(float(df_eval["hi_quality_score"].mean()), 1),
        "avg_telugu_quality_score": round(float(df_eval["te_quality_score"].mean()), 1),
        "hindi_validity_rate_%": round(float(df_eval["hi_is_valid"].mean() * 100), 1),
        "telugu_validity_rate_%": round(float(df_eval["te_is_valid"].mean() * 100), 1),
        "detailed_results": results_rows,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)

    logger.info("Multilingual evaluation saved to %s and %s", csv_path, json_path)
    return df_eval


if __name__ == "__main__":
    df = run_multilingual_benchmark()
    print("\n=== Multilingual Evaluation Summary ===")
    print(df[["category", "en_words", "hi_words", "hi_quality_score", "te_words", "te_quality_score"]].to_string(index=False))

