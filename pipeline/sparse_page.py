# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:50:39 2026

@author: Gaurav
"""

#!/usr/bin/env python3
import cv2, numpy as np, shutil, sys
from pathlib import Path

def main():
    print("🧠 Smart Page Sorter (Blank / Sparse / Content)")
    
    # 1. Paths
    src_input = input("📁 Source folder path: ").strip().strip('"').strip("'")
    dst_root = input("📂 Destination base folder (will create 3 subfolders): ").strip().strip('"').strip("'")

    # 2. Configuration
    try:
        threshold_val = int(input("🌑 Threshold for 'ink' (0-255) [default 200]: ").strip() or 200)
        if not (0 < threshold_val <= 255): raise ValueError("Must be 1-255.")
        
        # These define your 3 buckets
        max_blank_ratio = float(input("📉 Max ratio (%) for Blank [default 0.05]: ").strip() or 0.05)
        max_sparse_ratio = float(input("📉 Max ratio (%) for Sparse [default 0.8]: ").strip() or 0.8)
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}"); sys.exit(1)

    src_path = Path(src_input)
    base_dest = Path(dst_root)

    if not src_path.exists():
        print("❌ Source not found."); sys.exit(1)

    # Auto-create subfolders
    dir_content = base_dest / "01_Ready_for_Training"
    dir_sparse  = base_dest / "02_Review_Sparse"
    dir_blank   = base_dest / "03_Discard_Blank"
    
    for d in [dir_content, dir_sparse, dir_blank]:
        d.mkdir(parents=True, exist_ok=True)

    # Gather & Sort Images
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = sorted([f for f in src_path.iterdir() if f.suffix.lower() in valid_ext])
    if not files:
        print("⚠️ No images found."); sys.exit(0)

    print(f"\n📊 Sorting {len(files)} pages into 3 categories...\n")
    counters = {"content": 0, "sparse": 0, "blank": 0, "errors": 0}

    for i, img_path in enumerate(files, 1):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                counters["errors"] += 1
                continue

            # 1. Pre-processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Invert: Text becomes White (255), Background becomes Black (0)
            _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
            
            # 2. Morphological operations to connect fragmented letters into words
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary, kernel, iterations=2)
            
            # 3. Find Contours (Text Blobs)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter noise (blobs smaller than 10 pixels)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
            num_words = len(valid_contours)
            
            # 4. Calculate Dark Pixel Ratio
            total_pixels = img.shape[0] * img.shape[1]
            dark_pixels = cv2.countNonZero(binary)
            dark_pct = (dark_pixels / total_pixels) * 100

            # 5. Classification Logic
            # BLANK: Almost zero black pixels AND zero text components
            if dark_pct <= max_blank_ratio and num_words < 3:
                target_dir = dir_blank
                label = "⬜ Blank"
                counters["blank"] += 1
                
            # SPARSE: Low ink ratio OR very few word-blobs (like the example page)
            elif dark_pct <= max_sparse_ratio or num_words < 15:
                target_dir = dir_sparse
                label = "⚠️ Sparse"
                counters["sparse"] += 1
                
            # CONTENT: High ink density and many text components
            else:
                target_dir = dir_content
                label = "✅ Content"
                counters["content"] += 1

            # Move file
            shutil.copy2(str(img_path), target_dir / img_path.name)
            print(f"[{i}/{len(files)}] {label}: {img_path.name} (Ink:{dark_pct:.2f}%, Words:{num_words})")

        except Exception as e:
            counters["errors"] += 1
            print(f"[{i}/{len(files)}] ❌ Error {img_path.name}: {e}")

    print("\n📊 Sorting Complete!")
    print(f"✅ Content (01): {counters['content']}")
    print(f"⚠️ Sparse (02): {counters['sparse']}")
    print(f"⬜ Blank (03): {counters['blank']}")
    print(f"❌ Errors: {counters['errors']}")
    print(f"\n📂 Results saved to: {base_dest.absolute()}")

if __name__ == '__main__':
    main()