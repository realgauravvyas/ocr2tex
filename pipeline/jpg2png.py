# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:12:16 2026

@author: Gaurav
"""

#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image

def main():
    print("🔄 JPG → PNG Batch Converter")
    src_input = input("📁 Enter source folder path: ").strip().strip('"').strip("'")
    dst_input = input("📁 Enter destination folder path: ").strip().strip('"').strip("'")

    src_path = Path(src_input)
    dst_path = Path(dst_input)

    if not src_path.exists():
        print("❌ Error: Source folder does not exist.")
        sys.exit(1)
    if not src_path.is_dir():
        print("❌ Error: Source path is not a directory.")
        sys.exit(1)

    # Create destination if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)

    # Gather JPG/JPEG files
    extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    files = [f for f in src_path.iterdir() if f.suffix in extensions]

    if not files:
        print("⚠️  No JPG/JPEG files found in the source folder.")
        return

    success_count = 0
    fail_count = 0

    for i, img_path in enumerate(files, 1):
        try:
            with Image.open(img_path) as img:
                # Convert CMYK to RGB (PNG doesn't support CMYK well)
                if img.mode == 'CMYK':
                    img = img.convert('RGB')
                # Preserve transparency if present (RGBA/LA)
                elif img.mode not in ('RGB', 'RGBA', 'LA'):
                    img = img.convert('RGB')

                out_path = dst_path / f"{img_path.stem}.png"
                # optimize=True reduces file size without losing quality
                img.save(out_path, 'PNG', optimize=True)
                success_count += 1
                print(f"[{success_count + fail_count}/{len(files)}] ✅ {img_path.name}")
        except Exception as e:
            fail_count += 1
            print(f"[{success_count + fail_count}/{len(files)}] ❌ Failed {img_path.name}: {e}")

    print("\n📊 Conversion Complete!")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📂 Output saved to: {dst_path.absolute()}")

if __name__ == '__main__':
    main()