"""Run v4.1 on all 67 detected printed-only pages and report whether each
produces a blank output (the fix) or still transcribes (text)."""
import os, re, json, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, r"D:\ocr2tex\dashboard")
import torch
from PIL import Image
from benchmark_glm_ocr import load_model, generate_batch, cap_image_tokens
from transformers import AutoProcessor

V41 = r"D:\ocr2tex\output\glm-ocr-math-v4.1\final"
IMG = Path(r"D:\ocr2tex\workspace\9_split\images")
CLASS = r"D:\ocr2tex\workspace\handwriting_classification.json"
SPLITS = Path(r"D:\ocr2tex\workspace\9_split")

printed = json.load(open(CLASS, encoding="utf-8"))["no_pages"]
# which split each printed page is in (train pages were trained-as-blank; val/test are held out)
split_of = {}
for s in ("train", "val", "test"):
    for line in (SPLITS / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            split_of[json.loads(line)["id"]] = s

print(f"checking {len(printed)} printed pages with v4.1...", flush=True)
proc = AutoProcessor.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
cap_image_tokens(proc)
model = load_model(V41, "cuda")

results = {}
BS = 4
for i in range(0, len(printed), BS):
    chunk = printed[i:i + BS]
    imgs = [Image.open(IMG / f"{p}.png").convert("RGB") for p in chunk]
    texts, _ = generate_batch(model, proc, imgs, 1536, "cuda")
    for p, t in zip(chunk, texts):
        results[p] = len(t.strip())
    print(f"  {i+len(chunk)}/{len(printed)} done", flush=True)

blank = [p for p, n in results.items() if n == 0]
nonblank = sorted([(p, results[p]) for p in results if results[p] > 0], key=lambda x: -x[1])
print("\n================ v4.1 ON 67 PRINTED PAGES ================")
print(f"BLANK (0 chars): {len(blank)} / {len(printed)}")
print(f"NON-BLANK (still produced text): {len(nonblank)} / {len(printed)}")
if nonblank:
    print("\npages where v4.1 still generated output:")
    for p, n in nonblank:
        print(f"  {p} [{split_of.get(p,'?')}]: {n} chars")
# held-out generalization breakdown
held = [p for p in printed if split_of.get(p) in ("val", "test")]
held_blank = [p for p in held if results.get(p) == 0]
print(f"\nheld-out (val+test) printed pages: {len(held_blank)}/{len(held)} blank  (true generalization)")
json.dump({"blank": blank, "nonblank": dict(nonblank), "char_counts": results,
           "splits": {p: split_of.get(p) for p in printed}},
          open(r"D:\ocr2tex\workspace\v41_printed_check.json", "w"), indent=2)
print("saved -> workspace/v41_printed_check.json")
