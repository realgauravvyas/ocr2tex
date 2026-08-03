# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:03:44 2026

@author: Gaurav
"""

#!/usr/bin/env python3
import cv2, numpy as np, shutil, sys
from pathlib import Path

def main():
    print("📄 Blank & Content Page Sorter")
    
    # 1. Paths
    src_input = input("📁 Source folder path: ").strip().strip('"').strip("'")
    dst_nonblank = input("📁 Destination for NON-BLANK pages: ").strip().strip('"').strip("'")
    dst_blank = input("📁 Destination for BLANK/WHITE pages: ").strip().strip('"').strip("'")
    
    # 2. Detection Parameters
    try:
        dark_thresh = int(input("🌑 Pixel threshold for 'ink' (0-255) [default 210]: ").strip() or 210)
        max_dark_pct = float(input("📊 Max % of dark pixels to call 'blank' [default 0.05]: ").strip() or 0.05)
        if not (0 < dark_thresh <= 255):
            raise ValueError("Threshold must be 1-255.")
        if max_dark_pct < 0:
            raise ValueError("Percentage must be >= 0.")
    except ValueError as e:
        print(f"❌ Invalid input: {e}"); sys.exit(1)

    # Validate paths
    src_path = Path(src_input)
    dst_nb_path = Path(dst_nonblank)
    dst_b_path = Path(dst_blank)

    if not src_path.exists() or not src_path.is_dir():
        print("❌ Source folder does not exist or is not a directory."); sys.exit(1)

    # Safety: prevent destinations from overlapping with source
    src_resolved = src_path.resolve()
    for dst in [dst_nb_path, dst_b_path]:
        if dst.resolve() == src_resolved:
            print("❌ Destination cannot be the same as source."); sys.exit(1)
        dst.mkdir(parents=True, exist_ok=True)

    # Gather & sort images for deterministic order
    valid_ext = {'.png', '.jpg', '.jpeg'}
    files = sorted([f for f in src_path.iterdir() if f.suffix.lower() in valid_ext])
    if not files:
        print("⚠️  No supported image files found."); sys.exit(0)

    print(f"\n🔍 Sorting {len(files)} images (thresh={dark_thresh}, max_blank={max_dark_pct}%)...\n")
    kept_nb = kept_b = failed = 0

    for i, img_path in enumerate(files, 1):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            dark_count = np.sum(gray < dark_thresh)
            dark_pct = (dark_count / (h * w)) * 100

            if dark_pct <= max_dark_pct:
                target_dir = dst_b_path
                label = "⬜ BLANK"
                kept_b += 1
            else:
                target_dir = dst_nb_path
                label = "🖼️  CONTENT"
                kept_nb += 1

            shutil.copy2(str(img_path), target_dir / img_path.name)
            print(f"[{i}/{len(files)}] {label}: {img_path.name} ({dark_pct:.3f}%)")

        except Exception as e:
            failed += 1
            print(f"[{i}/{len(files)}] ❌ Error {img_path.name}: {e}")

    print("\n📊 Sorting Complete!")
    print(f"🖼️  Non-Blank/Content: {kept_nb} → {dst_nb_path.absolute()}")
    print(f"⬜ Blank/White: {kept_b} → {dst_b_path.absolute()}")
    print(f"❌ Errors/Corrupt: {failed}")

if __name__ == '__main__':
    main()