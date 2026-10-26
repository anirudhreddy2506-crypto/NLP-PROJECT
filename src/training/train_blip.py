"""
Reproducible fine-tuning pipeline for BLIP image captioning models.
Supports Apple Silicon MPS acceleration, gradient accumulation, checkpointing, and validation logging.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, get_linear_schedule_with_warmup

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.device import get_optimal_device
from src.utils.logger import setup_logger
from src.training.dataset import (
    ImageCaptionDataset,
    split_image_caption_data,
    prepare_smoke_test_data,
)

logger = setup_logger("visioncaption.training")


def train_blip(
    data_path: Optional[str] = None,
    output_dir: str = "models/fine_tuned",
    checkpoint_dir: str = "models/checkpoints",
    model_name: str = "Salesforce/blip-image-captioning-base",
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    num_epochs: int = 2,
    max_length: int = 40,
    resume_from_checkpoint: Optional[str] = None,
    device_preference: str = "auto",
    smoke_test: bool = False,
    smoke_samples: int = 20,
) -> Dict[str, Any]:
    """
    Executes fine-tuning loop for BLIP on image-caption pairs.

    Returns:
        Dict[str, Any]: Training summary including final train loss, val loss, and checkpoint paths.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Device resolution
    device = get_optimal_device(preference=device_preference)
    logger.info("Initializing training on device: %s", device)

    # 2. Data preparation
    if smoke_test or not data_path or not os.path.exists(data_path):
        logger.info("Preparing smoke-test dataset (%d samples)...", smoke_samples)
        data_path = prepare_smoke_test_data(
            output_dir=os.path.join(PROJECT_ROOT, "data/processed/smoke_test"),
            num_samples=smoke_samples,
        )

    with open(data_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    logger.info("Loaded %d raw image records from %s.", len(raw_records), data_path)

    # Split strictly by image ID
    train_records, val_records, test_records = split_image_caption_data(
        raw_records, train_ratio=0.70, val_ratio=0.15, seed=42
    )

    # Save test split for held-out evaluation
    test_split_path = os.path.join(PROJECT_ROOT, "data/processed/test_split.json")
    os.makedirs(os.path.dirname(test_split_path), exist_ok=True)
    with open(test_split_path, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)
    logger.info("Held-out test split (%d images) saved to %s.", len(test_records), test_split_path)

    # 3. Model & Processor Loading
    start_model = resume_from_checkpoint if resume_from_checkpoint else model_name
    logger.info("Loading BLIP processor from '%s'...", model_name)
    processor = BlipProcessor.from_pretrained(model_name)

    logger.info("Loading BLIP model weights from '%s'...", start_model)
    try:
        model = BlipForConditionalGeneration.from_pretrained(start_model)
        model.to(device)
    except Exception as e:
        if device.type == "mps":
            logger.warning("MPS loading failed (%s). Falling back to CPU.", e)
            device = torch.device("cpu")
            model = BlipForConditionalGeneration.from_pretrained(start_model)
            model.to(device)
        else:
            raise e

    # 4. Datasets and DataLoaders
    train_dataset = ImageCaptionDataset(train_records, processor=processor, max_length=max_length)
    val_dataset = ImageCaptionDataset(val_records, processor=processor, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info("Train batches: %d | Validation batches: %d", len(train_loader), len(val_loader))

    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps)
    )

    # 6. Training Loop
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        logger.info("--- Starting Epoch %d/%d ---", epoch, num_epochs)

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                total_train_loss += outputs.loss.item()
                step_count += 1

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            except Exception as e:
                if device.type == "mps":
                    logger.warning("MPS out-of-memory or kernel error during step %d: %s. Falling back to CPU.", step, e)
                    device = torch.device("cpu")
                    model.to(device)
                    # Retry current step on CPU
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    pixel_values = pixel_values.to(device)
                    labels = labels.to(device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                    loss.backward()
                    total_train_loss += outputs.loss.item()
                    step_count += 1
                else:
                    raise e

        avg_train_loss = total_train_loss / max(1, step_count)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_step_count = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()
                val_step_count += 1

        avg_val_loss = total_val_loss / max(1, val_step_count)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["epoch_times"].append(round(epoch_time, 2))

        logger.info(
            "Epoch %d/%d Complete | Train Loss: %.4f | Val Loss: %.4f | Duration: %.2fs",
            epoch, num_epochs, avg_train_loss, avg_val_loss, epoch_time
        )

        # Save intermediate epoch checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        processor.save_pretrained(ckpt_path)
        logger.info("Saved epoch checkpoint to %s.", ckpt_path)

    # 7. Final Model Export
    logger.info("Saving final fine-tuned model and processor to %s...", output_dir)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    total_duration = time.time() - start_time
    summary = {
        "base_model": model_name,
        "fine_tuned_dir": output_dir,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "history": history,
        "total_training_duration_seconds": round(total_duration, 2),
        "device_used": str(device),
        "test_split_path": test_split_path,
    }

    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training finished in %.2fs. Metrics saved to %s.", total_duration, metrics_file)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BLIP on image-caption dataset.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to JSON dataset.")
    parser.add_argument("--output_dir", type=str, default="models/fine_tuned", help="Output directory for final model.")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Directory for epoch checkpoints.")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base", help="Hugging Face base model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path.")
    parser.add_argument("--smoke_test", action="store_true", help="Run with smoke test dataset.")
    parser.add_argument("--smoke_samples", type=int, default=15, help="Number of smoke test images.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_blip(
        data_path=args.data_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        resume_from_checkpoint=args.resume,
        smoke_test=args.smoke_test,
        smoke_samples=args.smoke_samples,
    )
