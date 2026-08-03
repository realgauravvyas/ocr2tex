"""
Step 3: Build Dataset
Combines images and LaTeX annotations into the JSONL format
required for GLM-OCR fine-tuning.

Note: the training script (06) builds its own prompt via the processor chat
template and only reads the "assistant" (LaTeX) content from each sample.
The system/user fields here are stored for reference/portability.
"""

import json
import argparse
from pathlib import Path


def build_dataset(images_dir: str, annotations_dir: str, output_file: str, 
                  system_prompt_path: str):
    """
    Build JSONL dataset from images and annotations.

    Format per line:
    {
        "id": "page_00001",
        "image": "images/page_00001.png",
        "conversations": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "OCR this handwritten math page..."},
            {"role": "assistant", "content": "\\documentclass..."}
        ]
    }
    """
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load system prompt
    system_prompt = Path(system_prompt_path).read_text(encoding='utf-8').strip()
    
    # User prompt (consistent across all samples - matches training inference prompt)
    user_prompt = (
        "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
        "content into a complete, compilable LaTeX document. Ignore printed text, "
        "student info, page numbers, cancelled work and rough work. Output only LaTeX."
    )
    
    # Find all annotation files
    annotation_files = sorted(annotations_path.glob("*.tex"))
    
    print(f"Found {len(annotation_files)} annotations")
    
    samples = []
    skipped = 0
    
    for ann_file in annotation_files:
        image_name = f"{ann_file.stem}.png"
        image_file = images_path / image_name
        
        # Verify image exists
        if not image_file.exists():
            print(f"  WARNING: Image not found for {ann_file.name}, skipping")
            skipped += 1
            continue
        
        # Read annotation
        latex_content = ann_file.read_text(encoding='utf-8').strip()
        
        # Skip empty or template-only annotations
        if not latex_content or "YOUR TRANSCRIPTION BELOW" in latex_content:
            print(f"  WARNING: Empty/template annotation for {ann_file.name}, skipping")
            skipped += 1
            continue
        
        # Skip too-short annotations (likely empty pages or model refusals)
        # A real solution should be at least ~200 chars of content beyond boilerplate
        if len(latex_content) < 200:
            print(f"  WARNING: Too short ({len(latex_content)} chars) for {ann_file.name}, skipping")
            skipped += 1
            continue
        
        # Skip model refusals that slipped through
        refusal_phrases = ["please provide the image", "no handwritten content", 
                          "please attach", "i cannot see"]
        if any(p in latex_content.lower() for p in refusal_phrases):
            print(f"  WARNING: Model refusal in {ann_file.name}, skipping")
            skipped += 1
            continue
        
        # Remove review markers
        latex_content = latex_content.replace("% REVIEWED: OK", "").strip()
        
        # Build sample
        sample = {
            "id": ann_file.stem,
            "image": f"images/{image_name}",
            "conversations": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": latex_content
                }
            ]
        }
        
        samples.append(sample)
    
    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nDataset built successfully!")
    print(f"  Total samples: {len(samples)}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {output_path}")
    
    # Print stats
    if samples:
        latex_lengths = [len(s['conversations'][2]['content']) for s in samples]
        print(f"\n  LaTeX length stats:")
        print(f"    Min: {min(latex_lengths)} chars")
        print(f"    Max: {max(latex_lengths)} chars")
        print(f"    Avg: {sum(latex_lengths)//len(latex_lengths)} chars")
    
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build JSONL dataset for GLM-4V fine-tuning")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=r"d:\Kiro\dataset\images",
        help="Directory containing prepared images"
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default=r"d:\Kiro\dataset\annotations",
        help="Directory containing LaTeX annotations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"d:\Kiro\dataset\full_dataset.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=r"d:\Kiro\config\system_prompt.txt",
        help="Path to system prompt file"
    )
    
    args = parser.parse_args()
    build_dataset(args.images_dir, args.annotations_dir, args.output, args.system_prompt)
