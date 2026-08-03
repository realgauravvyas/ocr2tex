"""Build the v4.1 dataset: same train/val/test split as 9_split, but the
fully-printed pages (detected via MiniMax) get an EMPTY assistant target so the
model learns "printed page -> blank output" natively.

Splits are preserved exactly (a printed page stays in whichever split it was in)
-> no mixing, no leakage. Images are shared with 9_split via --image-base at
train time, so nothing is copied.
"""
import json
from pathlib import Path

SRC = Path(r"D:\ocr2tex\workspace\9_split")
OUT = Path(r"D:\ocr2tex\workspace\9_split_v4.1")
CLASS = Path(r"D:\ocr2tex\workspace\handwriting_classification.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    printed = set(json.loads(CLASS.read_text(encoding="utf-8"))["no_pages"])
    print(f"printed-only pages to blank: {len(printed)}")
    summary = {}
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in (SRC / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        blanked = 0
        for r in rows:
            if r["id"] in printed:
                for c in r["conversations"]:
                    if c["role"] == "assistant":
                        c["content"] = ""  # blank target for a fully-printed page
                blanked += 1
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = (len(rows), blanked)
        print(f"  {split}: {len(rows)} samples, {blanked} blanked")
    # sanity: no id appears in more than one split (no leakage)
    ids = {}
    dup = 0
    for split in ("train", "val", "test"):
        for l in (OUT / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            i = json.loads(l)["id"]
            if i in ids:
                dup += 1
            ids[i] = split
    print(f"total unique ids: {len(ids)} | cross-split duplicates: {dup} (must be 0)")
    print(f"written -> {OUT}  (use --image-base {SRC} so images resolve)")


if __name__ == "__main__":
    main()
