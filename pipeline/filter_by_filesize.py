# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:06:10 2026

@author: Gaurav
"""

#!/usr/bin/env python3
import shutil, sys, os
from pathlib import Path

def main():
    print("📏 File-Size Page Sorter")
    print("💡 Logic: Handwriting breaks compression smoothness. Inked pages = higher KB.")
    print("-" * 50)
    
    # 1. Paths
    src_input = input("📁 Source Folder: ").strip().strip('"').strip("'")
    dst_input = input("📂 Destination Folder: ").strip().strip('"').strip("'")

    # 2. Threshold Configuration
    try:
        threshold_kb = float(input("📐 Size Threshold (KB) [default 100]: ").strip() or "100")
        if threshold_kb <= 0:
            raise ValueError
        threshold_bytes = threshold_kb * 1024
    except ValueError:
        print("❌ Invalid threshold. Please enter a positive number."); sys.exit(1)

    src_path = Path(src_input)
    dst_path = Path(dst_input)

    if not src_path.exists() or not src_path.is_dir():
        print("❌ Source folder not found or is not a directory."); sys.exit(1)

    # Auto-create categorized subfolders
    dir_blanks = dst_path / "probable_blanks"
    dir_content = dst_path / "probable_content"
    dir_blanks.mkdir(parents=True, exist_ok=True)
    dir_content.mkdir(parents=True, exist_ok=True)

    # 3. Process Files
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    files = sorted([f for f in src_path.iterdir() if f.suffix.lower() in valid_ext])
    
    if not files:
        print("⚠️ No supported image files found."); sys.exit(0)

    print(f"\n🔄 Scanning {len(files)} files (Threshold: {threshold_kb} KB)...\n")
    counts = {"blank": 0, "content": 0, "error": 0}

    for i, f in enumerate(files, 1):
        try:
            size_bytes = f.stat().st_size
            size_kb = size_bytes / 1024.0

            # Route based on threshold
            if size_bytes < threshold_bytes:
                target = dir_blanks / f.name
                counts["blank"] += 1
                tag = "⬜ Probable Blank"
            else:
                target = dir_content / f.name
                counts["content"] += 1
                tag = "✅ Probable Content"

            # Copy to destination
            shutil.copy2(str(f), str(target))
            print(f"[{i}/{len(files)}] {tag}: {f.name} ({size_kb:.1f} KB)")

        except Exception as e:
            counts["error"] += 1
            print(f"[{i}/{len(files)}] ❌ Error: {f.name} - {e}")

    # 4. Summary
    print("\n📊 Sorting Complete!")
    print(f"⬜ Probable Blanks: {counts['blank']}")
    print(f"✅ Probable Content: {counts['content']}")
    print(f"❌ Errors: {counts['error']}")
    print(f"📂 Results saved to: {dst_path.absolute()}")

if __name__ == '__main__':
    main()