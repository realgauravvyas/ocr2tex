"""Resize page images to the token budget GLM-OCR v3.1/v4.1 were fine-tuned at.

Both adapters were trained with max_image_tokens=1536. Each token covers a
14x14 patch with 2x2 merging, i.e. 28*28 = 784 pixels, so the budget is
roughly 1536 * 784 ~= 1.2 megapixels.

Feeding a full-resolution scan or phone photo (often 8-14 MP) pushes the model
far off-distribution -- in practice it degenerates into repeating the same line
until it hits the token cap. Front-ends differ here: llama.cpp's llama-mtmd-cli
downscales automatically, LM Studio does not, so resize before uploading there.

Usage:
    python resize_for_ocr.py page.jpg                 # -> page_ocr.png
    python resize_for_ocr.py in_dir/ -o out_dir/      # whole folder
"""

import argparse
from pathlib import Path

from PIL import Image

# 1536 image tokens * (14 patch * 2 merge)^2 pixels per token
MAX_PIXELS = 1536 * (14 * 2) ** 2  # 1,204,224

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def resize_one(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    pixels = w * h

    if pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / pixels) ** 0.5
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"  {src.name}: {w}x{h} ({pixels/1e6:.1f} MP) -> "
              f"{new_size[0]}x{new_size[1]} ({new_size[0]*new_size[1]/1e6:.2f} MP)")
    else:
        print(f"  {src.name}: {w}x{h} ({pixels/1e6:.2f} MP) already within budget")

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or directory")
    ap.add_argument("-o", "--out", default=None,
                    help="output file or directory (default: alongside input, '_ocr' suffix)")
    args = ap.parse_args()

    src = Path(args.input)

    if src.is_dir():
        out_dir = Path(args.out) if args.out else src.parent / f"{src.name}_ocr"
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
        if not files:
            print(f"No images found in {src}")
            return
        print(f"Resizing {len(files)} image(s) -> {out_dir}")
        for f in files:
            resize_one(f, out_dir / f"{f.stem}.png")
    else:
        dst = Path(args.out) if args.out else src.with_name(f"{src.stem}_ocr.png")
        resize_one(src, dst)
        print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
