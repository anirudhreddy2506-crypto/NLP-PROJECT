"""
Unit tests for VisionCaption AI training pipeline, schema inspection, image-level splitting, and checkpointing.
"""

import os
import json
import pytest
from PIL import Image
import torch
from transformers import BlipProcessor

from src.training.dataset import (
    inspect_schema,
    split_image_caption_data,
    prepare_smoke_test_data,
    ImageCaptionDataset,
)
from src.training.train_blip import train_blip
from src.training.evaluate import compute_metrics


def test_schema_inspection():
    """Verify dynamic schema detection correctly identifies columns."""
    coco_sample = {"image_id": 1234, "image_path": "path/to/img.jpg", "captions": ["a cat", "a kitten"]}
    schema = inspect_schema(coco_sample)
    assert schema["image_field"] == "image_path"
    assert schema["caption_field"] == "captions"
    assert schema["id_field"] == "image_id"

    custom_sample = {"img": "cat.png", "text": "a cute cat"}
    schema2 = inspect_schema(custom_sample)
    assert schema2["image_field"] == "img"
    assert schema2["caption_field"] == "text"


def test_image_level_splitting_no_leakage():
    """Verify that images are never split across train, val, and test partitions."""
    dummy_records = [
        {"image_id": f"img_{i}", "image_path": f"img_{i}.jpg", "captions": [f"cap {i}_1", f"cap {i}_2"]}
        for i in range(20)
    ]

    train, val, test = split_image_caption_data(dummy_records, train_ratio=0.7, val_ratio=0.15, seed=123)

    train_ids = {r["image_id"] for r in train}
    val_ids = {r["image_id"] for r in val}
    test_ids = {r["image_id"] for r in test}

    # Strict disjoint assertion
    assert train_ids.isdisjoint(val_ids), "Train and Val share image IDs!"
    assert train_ids.isdisjoint(test_ids), "Train and Test share image IDs!"
    assert val_ids.isdisjoint(test_ids), "Val and Test share image IDs!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == 20


def test_dataset_padding_masking():
    """Verify that pad tokens in labels are converted to -100 to ignore padding in cross-entropy loss."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    sample_records = [
        {"image_id": "test_1", "image": Image.new("RGB", (64, 64)), "captions": ["short"]}
    ]
    ds = ImageCaptionDataset(sample_records, processor=processor, max_length=20)
    item = ds[0]

    assert "pixel_values" in item
    assert "input_ids" in item
    assert "labels" in item
    # Check that padded positions in labels are -100
    pad_id = processor.tokenizer.pad_token_id
    pad_positions = (item["input_ids"] == pad_id).nonzero(as_tuple=True)[0]
    for pos in pad_positions:
        assert item["labels"][pos].item() == -100


def test_compute_metrics_bleu_rouge():
    """Verify BLEU and ROUGE metric calculation with multiple reference captions."""
    predictions = ["a red apple on a white plate", "a blue car on the road"]
    references = [
        ["a red apple sitting on a ceramic plate", "an apple on the table"],
        ["a blue vehicle on the highway", "a blue sports car parked outside"],
    ]
    metrics = compute_metrics(predictions, references)
    assert "BLEU-1" in metrics
    assert "BLEU-4" in metrics
    assert "ROUGE-L" in metrics
    assert 0.0 <= metrics["BLEU-1"] <= 1.0
    assert 0.0 <= metrics["ROUGE-L"] <= 1.0


@pytest.mark.slow
def test_train_blip_smoke_run(tmp_path):
    """Verify that a minimal training run executes forward/backward pass and outputs checkpoints."""
    out_dir = str(tmp_path / "fine_tuned")
    ckpt_dir = str(tmp_path / "checkpoints")

    summary = train_blip(
        output_dir=out_dir,
        checkpoint_dir=ckpt_dir,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_epochs=1,
        smoke_test=True,
        smoke_samples=4,
    )

    assert summary["final_train_loss"] is not None
    assert summary["final_val_loss"] is not None
    assert os.path.exists(os.path.join(out_dir, "training_metrics.json"))
