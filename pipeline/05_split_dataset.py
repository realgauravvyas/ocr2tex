"""
Step 5: Split Dataset
Splits the full dataset into train/val/test sets.
"""

import json
import random
import argparse
from pathlib import Path


def split_dataset(input_file: str, output_dir: str, 
                  train_ratio: float = 0.83, val_ratio: float = 0.10,
                  seed: int = 42):
    """
    Split dataset into train/val/test.
    
    Args:
        input_file: Path to full_dataset.jsonl
        output_dir: Directory to save split files
        train_ratio: Fraction for training (default 0.83)
        val_ratio: Fraction for validation (default 0.10)
        seed: Random seed for reproducibility
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read all samples
    with open(input_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]
    
    total = len(samples)
    print(f"Total samples: {total}")
    
    # Shuffle
    random.seed(seed)
    random.shuffle(samples)
    
    # Calculate split sizes
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    # Split
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]
    
    # Write splits
    splits = {
        'train.jsonl': train_samples,
        'val.jsonl': val_samples,
        'test.jsonl': test_samples,
    }
    
    for filename, data in splits.items():
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"  {filename}: {len(data)} samples")
    
    print(f"\nSplit complete!")
    print(f"  Train: {train_size} ({train_size/total*100:.1f}%)")
    print(f"  Val:   {val_size} ({val_size/total*100:.1f}%)")
    print(f"  Test:  {test_size} ({test_size/total*100:.1f}%)")
    print(f"\nFiles saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument(
        "--input",
        type=str,
        default=r"d:\Kiro\dataset\full_dataset.jsonl",
        help="Path to full dataset JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"d:\Kiro\dataset",
        help="Output directory for split files"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Training set ratio (default: 0.80)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation set ratio (default: 0.10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        print("ERROR: train_ratio + val_ratio must be <= 1.0")
        exit(1)
    
    split_dataset(args.input, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
