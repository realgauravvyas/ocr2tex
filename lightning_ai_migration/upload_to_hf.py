# -*- coding: utf-8 -*-
"""
Upload dataset directly to a private Hugging Face Dataset repository.
Lightning AI can pull from Hugging Face at 300+ MB/s in under 60 seconds!

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  (or set HF_TOKEN environment variable)
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = input("Enter your Hugging Face private dataset name (e.g. your-username/math-ocr-v5): ").strip()
if not REPO_ID:
    print("No repo specified. Exiting.")
    sys.exit(1)

api = HfApi()
print(f"Creating / verifying private repository: {REPO_ID} ...")
api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)

SPLIT_DIR = Path(r"D:\ocr2tex\workspace\9_split_v5")
IMAGE_BASE = Path(r"D:\ocr2tex\workspace\9_split\images")

# 1. Upload splits
print("Uploading train.jsonl, val.jsonl, test.jsonl...")
for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    p = SPLIT_DIR / f
    if p.exists():
        print(f"  Uploading {f}...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f,
            repo_id=REPO_ID,
            repo_type="dataset",
        )

# 2. Upload images folder
print("Uploading images folder (10.5 GB, 14,000 files with chunked resume)...")
api.upload_folder(
    folder_path=str(IMAGE_BASE),
    path_in_repo="images",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"\n[SUCCESS] Upload complete! In Lightning AI Studio, you can download it via:")
print(f"  git clone https://huggingface.co/datasets/{REPO_ID} data")
