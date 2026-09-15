# -*- coding: utf-8 -*-
"""
Builds glm_ocr_v5_package.zip for Google Colab.

Contents:
    adapter/           <- the fine-tuned GLM-OCR v5.0 LoRA adapter weights & configs
    test_manifest.json <- ground-truth & metadata for all 700 held-out test pages
    test_images/       <- images for all 700 held-out test pages (page 1 to 700)
"""

import os
import json
import zipfile
from pathlib import Path

BASE = Path(r"D:\ocr2tex")
ADAPTER_DIR = BASE / "colab" / "glm_ocr_v5_extracted" / "output" / "final"
TEST_SPLIT = BASE / "workspace" / "9_split_v5" / "test.jsonl"
IMAGES_DIR = BASE / "workspace" / "9_split" / "images"
OUT_ZIP = BASE / "colab" / "glm_ocr_v5_package.zip"
MANIFEST_FILE = BASE / "colab" / "test_manifest.json"


def main():
    assert ADAPTER_DIR.exists(), f"Adapter directory not found: {ADAPTER_DIR}"
    assert TEST_SPLIT.exists(), f"Test split not found: {TEST_SPLIT}"
    assert IMAGES_DIR.exists(), f"Images directory not found: {IMAGES_DIR}"

    print(f"Opening {OUT_ZIP} for writing...")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # 1. Adapter files
        print("\nAdding adapter files:")
        for f in sorted(ADAPTER_DIR.iterdir()):
            if f.is_file():
                z.write(f, f"adapter/{f.name}")
                print(f"  + adapter/{f.name} ({f.stat().st_size / 1e6:.2f} MB)")

        # 2. Test manifest
        if MANIFEST_FILE.exists():
            z.write(MANIFEST_FILE, "test_manifest.json")
            print(f"\nAdded test_manifest.json ({MANIFEST_FILE.stat().st_size / 1e6:.2f} MB)")

        # 3. 700 Test images
        print("\nAdding 700 test images (this enables testing any page 1 to 700)...")
        with open(TEST_SPLIT, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                sid = item["id"]
                img_path = IMAGES_DIR / f"{sid}.png"
                if img_path.exists():
                    z.write(img_path, f"test_images/{sid}.png")
                    count += 1
                else:
                    print(f"  [WARN] Missing image: {img_path}")
            print(f"  Added {count} test images to test_images/")

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"\n=======================================================")
    print(f"Successfully created: {OUT_ZIP}")
    print(f"Package Size: {size_mb:.1f} MB")
    print("=======================================================")
    print("Next step: Upload this zip file to your Google Drive,")
    print("set sharing to 'Anyone with the link can view',")
    print("and copy the file ID into GLM_ocr_v5_Demo.ipynb!")


if __name__ == "__main__":
    main()
