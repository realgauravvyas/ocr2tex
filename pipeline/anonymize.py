# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:40:58 2026

@author: Gaurav
"""

#!/usr/bin/env python3
import cv2
import sys
from pathlib import Path

def get_paper_color(img, margin=40):
    """Estimate paper background color from bottom corners."""
    h, w = img.shape[:2]
    margin = min(margin, h // 4, w // 4)

    region_tl = img[-margin:, :margin]
    region_tr = img[-margin:, -margin:]

    gray_avg = (
        cv2.cvtColor(region_tl, cv2.COLOR_BGR2GRAY).mean() +
        cv2.cvtColor(region_tr, cv2.COLOR_BGR2GRAY).mean()
    ) / 2.0

    bg_val = int(round(gray_avg))
    return (bg_val, bg_val, bg_val)

def main():
    print("🔒 Document Anonymizer (Top % Auto-Cover)")

    # INPUTS
    src_input = input("📁 Source folder path: ").strip().strip('"').strip("'")
    dst_input = input("📁 Destination folder path: ").strip().strip('"').strip("'")

    # Pages
    try:
        num_pages = int(input("🔢 Number of pages to process (0 = all): ").strip())
        if num_pages < 0:
            raise ValueError
    except:
        print("❌ Invalid page number")
        sys.exit(1)

    # Percentage
    try:
        pct = float(input("📏 Percentage of top to white out: ").strip())
        if not (0 < pct <= 100):
            raise ValueError
    except:
        print("❌ Invalid percentage")
        sys.exit(1)

    # PATH VALIDATION
    src_path = Path(src_input)
    if not src_path.exists():
        print("❌ Source folder not found")
        sys.exit(1)

    dst_path = Path(dst_input)
    dst_path.mkdir(parents=True, exist_ok=True)

    # FILE LIST
    valid_ext = {'.png', '.jpg', '.jpeg'}
    files = sorted([
        f for f in src_path.iterdir()
        if f.suffix.lower() in valid_ext
    ])

    if not files:
        print("⚠️ No images found")
        sys.exit(0)

    if num_pages > 0:
        files = files[:num_pages]

    print(f"\n🔄 Processing {len(files)} images...\n")

    success, fail = 0, 0

    for i, img_path in enumerate(files, 1):
        try:
            img = cv2.imread(str(img_path))

            if img is None:
                raise ValueError("Image could not be read")

            h, w = img.shape[:2]
            crop_h = int(h * pct / 100)

            if crop_h <= 0:
                raise ValueError("Crop height too small")

            bg_color = get_paper_color(img)

            cv2.rectangle(img, (0, 0), (w, crop_h), bg_color, -1)

            out_path = dst_path / img_path.name

            # FIX: format-specific saving
            if img_path.suffix.lower() == ".png":
                ok = cv2.imwrite(str(out_path), img,
                                 [cv2.IMWRITE_PNG_COMPRESSION, 6])
            else:
                ok = cv2.imwrite(str(out_path), img,
                                 [cv2.IMWRITE_JPEG_QUALITY, 95])

            if not ok:
                raise IOError("Failed to write image")

            success += 1
            print(f"[{i}/{len(files)}] ✅ {img_path.name}")

        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] ❌ {img_path.name} → {e}")

    print("\n📊 DONE")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {fail}")
    print(f"📂 Output: {dst_path.resolve()}")

if __name__ == "__main__":
    main()

