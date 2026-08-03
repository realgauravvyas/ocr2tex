"""
Step 1: Prepare Images
Copies images from source directory, renames them with consistent naming,
and organizes them into the dataset/images folder.
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import argparse


def prepare_images(source_dir: str, output_dir: str, max_size: int = 2048):
    """
    Copy and organize images from source to dataset folder.
    
    Args:
        source_dir: Path to source images (D:\Open Code\pages for deskewing)
        output_dir: Path to output dataset/images folder
        max_size: Maximum dimension (width or height) for resizing. Set 0 to skip.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}
    
    # Collect all image files
    image_files = sorted([
        f for f in source_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        # Also check subdirectories
        image_files = sorted([
            f for f in source_path.rglob('*')
            if f.suffix.lower() in image_extensions
        ])
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    # Process and copy images
    manifest = []
    for idx, img_file in enumerate(image_files, start=1):
        # Consistent naming: page_XXXXX.png
        new_name = f"page_{idx:05d}.png"
        output_file = output_path / new_name
        
        try:
            # Open and optionally resize
            img = Image.open(img_file)
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (preserving aspect ratio)
            if max_size > 0:
                w, h = img.size
                if max(w, h) > max_size:
                    ratio = max_size / max(w, h)
                    new_w = int(w * ratio)
                    new_h = int(h * ratio)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Save as PNG
            img.save(output_file, 'PNG', quality=95)
            
            manifest.append({
                'id': f"page_{idx:05d}",
                'original_name': img_file.name,
                'new_name': new_name,
                'size': img.size
            })
            
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"  ERROR processing {img_file.name}: {e}")
            continue
    
    # Save manifest
    import json
    manifest_path = output_path.parent / "image_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDone! Processed {len(manifest)} images.")
    print(f"Images saved to: {output_path}")
    print(f"Manifest saved to: {manifest_path}")
    
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare images for dataset creation")
    parser.add_argument(
        "--source", 
        type=str, 
        default=r"D:\Open Code\pages for deskewing",
        help="Source directory containing deskewed images"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=r"d:\Kiro\dataset\images",
        help="Output directory for organized images"
    )
    parser.add_argument(
        "--max-size", 
        type=int, 
        default=0,
        help="Max image dimension (0 = no resize, keep original). GLM-4V handles 1120px internally."
    )
    
    args = parser.parse_args()
    prepare_images(args.source, args.output, args.max_size)
