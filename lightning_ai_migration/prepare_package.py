# -*- coding: utf-8 -*-
"""
Helper script to package code, checkpoint, and dataset for Lightning AI Studio.
Run this locally on your Windows machine:
    python prepare_package.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
SPLIT_DIR = Path(r"D:\ocr2tex\workspace\9_split_v5")
IMAGE_BASE = Path(r"D:\ocr2tex\workspace\9_split\images")
CKPT_DIR = Path(r"D:\ocr2tex\v5 training\output\checkpoint-250")

print("==========================================================")
print("GLM-OCR v5.0 - Lightning AI Migration Packaging Tool")
print("==========================================================")

def pack_checkpoint():
    if not CKPT_DIR.exists():
        print(f"[SKIP] Checkpoint not found at {CKPT_DIR}")
        return
    out_tar = BASE_DIR / "checkpoint-250.tar.gz"
    print(f"\n[1/3] Compressing checkpoint-250 (~160 MB)...")
    cmd = ["tar", "-czf", str(out_tar), "-C", str(CKPT_DIR.parent), CKPT_DIR.name]
    subprocess.run(cmd, check=True)
    print(f"       Saved: {out_tar.name} ({round(out_tar.stat().st_size / (1024**2), 1)} MB)")

def pack_code():
    out_tar = BASE_DIR / "code.tar.gz"
    print(f"\n[2/3] Compressing training code & scripts...")
    files = ["train_lightning.py", "requirements.txt", "run_train.sh"]
    cmd = ["tar", "-czf", str(out_tar), "-C", str(BASE_DIR)] + files
    subprocess.run(cmd, check=True)
    print(f"       Saved: {out_tar.name}")

def pack_dataset():
    temp_data = BASE_DIR / "temp_data_stage"
    temp_data.mkdir(parents=True, exist_ok=True)
    out_tar = BASE_DIR / "dataset.tar.gz"
    
    print(f"\n[3/3] Preparing dataset package (10.5 GB images + splits)...")
    print(f"      (This might take 3-5 minutes to compress 14,000 images)")
    
    # Copy jsonls
    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        src = SPLIT_DIR / f
        if src.exists():
            shutil.copy2(src, temp_data / f)

    # Use tar directly
    # In the archive we want:
    # ./data/train.jsonl
    # ./data/val.jsonl
    # ./data/test.jsonl
    # ./data/images/...
    print("      Creating dataset.tar.gz with tar...")
    
    # Stage directory structure
    stage_root = BASE_DIR / "data_stage"
    stage_data = stage_root / "data"
    stage_data.mkdir(parents=True, exist_ok=True)
    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        src = SPLIT_DIR / f
        if src.exists():
            shutil.copy2(src, stage_data / f)
            
    # To avoid duplicating 10.5GB, we can make directory junction or hardlink, or directly tar
    images_link = stage_data / "images"
    if not images_link.exists():
        try:
            # Create Windows directory junction (instant, 0 disk space)
            subprocess.run(["cmd", "/c", "mklink", "/J", str(images_link), str(IMAGE_BASE)], check=True)
            print("      Created instant junction for images (0 extra disk space consumed).")
        except Exception:
            pass

    cmd = ["tar", "-czf", str(out_tar), "-C", str(stage_root), "data"]
    subprocess.run(cmd, check=True)
    
    # Cleanup junction
    if images_link.exists():
        subprocess.run(["cmd", "/c", "rmdir", str(images_link)], check=False)
    shutil.rmtree(stage_root, ignore_errors=True)
    shutil.rmtree(temp_data, ignore_errors=True)
    
    print(f"\n[SUCCESS] Dataset archive created: {out_tar.name} ({round(out_tar.stat().st_size / (1024**3), 2)} GB)")

if __name__ == "__main__":
    pack_code()
    pack_checkpoint()
    pack_dataset()
    print("\n==========================================================")
    print("All packages ready in: " + str(BASE_DIR))
    print("==========================================================")
