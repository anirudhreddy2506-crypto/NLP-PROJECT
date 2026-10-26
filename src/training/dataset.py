"""
Dataset loading, schema inspection, image-level splitting, and PyTorch dataset wrapper for BLIP fine-tuning.
Guarantees zero data leakage by splitting strictly on unique image IDs.
"""

import os
import json
import random
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from transformers import BlipProcessor

logger = logging.getLogger(__name__)


def inspect_schema(sample_record: Dict[str, Any]) -> Dict[str, str]:
    """
    Dynamically identifies column keys in image-caption datasets.
    Handles MS COCO, Flickr8k, and custom formats.
    """
    keys = list(sample_record.keys())
    schema = {}

    # Identify image field
    for img_key in ["image", "img", "image_path", "filepath", "filename"]:
        if img_key in keys:
            schema["image_field"] = img_key
            break

    # Identify caption field
    for cap_key in ["caption", "captions", "sentences", "text", "description"]:
        if cap_key in keys:
            schema["caption_field"] = cap_key
            break

    # Identify image id field
    for id_key in ["image_id", "id", "img_id", "cocoid"]:
        if id_key in keys:
            schema["id_field"] = id_key
            break

    if "image_field" not in schema:
        raise KeyError(f"Could not identify image column in schema. Available keys: {keys}")
    if "caption_field" not in schema:
        raise KeyError(f"Could not identify caption column in schema. Available keys: {keys}")

    return schema


def split_image_caption_data(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits image-caption records strictly by IMAGE ID to eliminate reference caption leakage.

    Args:
        records: List of records containing image references and captions.
        train_ratio: Fraction of unique images for training.
        val_ratio: Fraction of unique images for validation.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_records, val_records, test_records).
    """
    if not records:
        return [], [], []

    schema = inspect_schema(records[0])
    id_field = schema.get("id_field", schema["image_field"])

    # Group records by unique image identifier
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        img_id = r.get(id_field)
        if isinstance(img_id, (dict, list)):
            img_id = str(img_id)
        grouped.setdefault(img_id, []).append(r)

    unique_ids = sorted(list(grouped.keys()), key=lambda x: str(x))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio)) if (n_total - n_train) > 1 else 0

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train : n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val :])

    # Ensure test has at least one image if total > 2
    if not test_ids and len(val_ids) > 1:
        moved = val_ids.pop()
        test_ids.add(moved)

    train_data = [item for img_id in train_ids for item in grouped[img_id]]
    val_data = [item for img_id in val_ids for item in grouped[img_id]]
    test_data = [item for img_id in test_ids for item in grouped[img_id]]

    logger.info(
        "Image-level split completed without leakage: %d train images (%d samples), %d val images (%d samples), %d test images (%d samples).",
        len(train_ids),
        len(train_data),
        len(val_ids),
        len(val_data),
        len(test_ids),
        len(test_data),
    )

    return train_data, val_data, test_data


def prepare_smoke_test_data(
    output_dir: str = "data/processed/smoke_test",
    num_samples: int = 30,
) -> str:
    """
    Generates a reproducible, lightweight dataset with diverse geometric and scene compositions
    and paired reference captions (with multiple references per image).
    Allows rapid fine-tuning verification without downloading multi-GB datasets.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    scenes = [
        {"bg": (135, 206, 235), "shape": "house", "color": (220, 20, 60),
         "captions": ["a red house with a green roof under a blue sky", "a small cozy red house with trees nearby", "a red cottage situated on a hill"]},
        {"bg": (245, 245, 245), "shape": "apple", "color": (200, 0, 0),
         "captions": ["a fresh red apple sitting on a ceramic plate", "a ripe red apple on a clean table surface", "a single red apple with a small green leaf"]},
        {"bg": (25, 25, 112), "shape": "moon", "color": (255, 255, 224),
         "captions": ["a bright glowing moon shining in the night sky", "the full moon illuminated in deep dark space", "a nighttime sky with a luminous moon"]},
        {"bg": (34, 139, 34), "shape": "flower", "color": (255, 105, 180),
         "captions": ["a vibrant pink flower blooming in a green garden", "a close up of a bright pink blossom", "a pink flower surrounded by lush green foliage"]},
        {"bg": (230, 240, 255), "shape": "car", "color": (30, 144, 255),
         "captions": ["a blue sports car parked along an open roadway", "a sleek blue sedan sitting outdoors", "a bright blue automobile in daylight"]},
    ]

    records = []
    for i in range(num_samples):
        scene = scenes[i % len(scenes)]
        img_id = f"smoke_img_{i:04d}"
        img_filename = f"{img_id}.png"
        img_path = os.path.join(images_dir, img_filename)

        # Generate synthetic image
        img = Image.new("RGB", (384, 384), color=scene["bg"])
        draw = ImageDraw.Draw(img)
        shape = scene["shape"]
        color = scene["color"]

        if shape == "house":
            draw.rectangle([100, 180, 280, 320], fill=color)
            draw.polygon([(80, 180), (190, 100), (300, 180)], fill=(34, 139, 34))
            draw.rectangle([160, 240, 220, 320], fill=(240, 240, 240))
        elif shape == "apple":
            draw.ellipse([100, 120, 280, 300], fill=color)
            draw.line([190, 120, 200, 80], fill=(34, 139, 34), width=6)
        elif shape == "moon":
            draw.ellipse([120, 120, 260, 260], fill=color)
        elif shape == "flower":
            for offset in [(0, -40), (40, 0), (0, 40), (-40, 0)]:
                draw.ellipse([150 + offset[0], 150 + offset[1], 230 + offset[0], 230 + offset[1]], fill=color)
            draw.ellipse([160, 160, 220, 220], fill=(255, 215, 0))
        elif shape == "car":
            draw.rectangle([80, 200, 300, 280], fill=color)
            draw.rectangle([120, 140, 260, 200], fill=(200, 220, 255))
            draw.ellipse([110, 260, 160, 310], fill=(30, 30, 30))
            draw.ellipse([220, 260, 270, 310], fill=(30, 30, 30))

        img.save(img_path, "PNG")

        # Each record contains multiple reference captions
        records.append({
            "image_id": img_id,
            "image_path": img_path,
            "captions": scene["captions"],
        })

    metadata_path = os.path.join(output_dir, "dataset.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Created smoke test dataset at %s with %d images.", output_dir, len(records))
    return metadata_path


class ImageCaptionDataset(Dataset):
    """
    PyTorch Dataset for BLIP fine-tuning.
    Handles images and reference captions with proper label padding masking (-100).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        processor: BlipProcessor,
        max_length: int = 40,
    ):
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        # Unroll multiple reference captions per image into (image, single_caption) pairs
        for record in records:
            img_ref = record.get("image_path") or record.get("image")
            captions = record.get("captions") or record.get("caption")

            if isinstance(captions, str):
                caption_list = [captions]
            elif isinstance(captions, list):
                caption_list = captions
            else:
                caption_list = [str(captions)]

            for cap in caption_list:
                self.samples.append({
                    "image": img_ref,
                    "caption": cap,
                    "image_id": record.get("image_id", "unknown"),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        image_val = item["image"]

        if isinstance(image_val, str):
            image = Image.open(image_val).convert("RGB")
        elif isinstance(image_val, Image.Image):
            image = image_val.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_val)}")

        caption = item["caption"]

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze batch dimension added by processor
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Labels for language modeling loss:
        # Ignore padding tokens in cross-entropy loss by setting them to -100
        labels = encoding["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        encoding["labels"] = labels

        return encoding
