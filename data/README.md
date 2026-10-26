# Data Directory

This directory holds datasets and splits used for fine-tuning and evaluation.
- `data/raw/`: Original downloaded images and annotations (ignored in git).
- `data/processed/`: Processed image-caption splits (train, validation, test) split strictly by image ID to prevent data leakage.
