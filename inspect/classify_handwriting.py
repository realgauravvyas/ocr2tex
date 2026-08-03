"""Classify dataset pages as handwritten vs printed-only, via MiniMax M3.

Pages that are entirely printed (e.g. a printed question sheet) or blank should
be NO. Used to scope/fix the label bug where MiniMax transcribed printed text on
pure-printed pages. Saves per-page verdicts so the result can drive a dataset fix.
"""
import os
import re
import sys
import json
import base64
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

IMG_DIR = Path(r"D:\ocr2tex\workspace\9_split\images")
OUT = Path(r"D:\ocr2tex\workspace\handwriting_classification.json")
CACHE = Path(r"D:\ocr2tex\workspace\handwriting_cache.jsonl")
API_BASE = "https://api.tokenrouter.com/v1"
MODEL = "MiniMax-M3"
KEYS = [k for k in os.environ.get("TOKENROUTER_API_KEYS", "").split(",") if k]

SYSTEM = "You are a precise document classifier for scanned university exam pages."
USER = (
    "Look at this scanned exam page. Does it contain ANY HANDWRITTEN content "
    "(handwritten mathematics, handwritten text, or handwritten working)?\n"
    "- If there is any handwriting at all, answer YES.\n"
    "- If the page is ENTIRELY printed/typed (e.g. a printed question sheet) "
    "or completely blank, answer NO.\n"
    "Reply with exactly one word: YES or NO."
)


def strip_think(t):
    t = re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in (t or "").lower():
        t = re.split(r"<think>", t, flags=re.IGNORECASE)[-1]
    return t


def classify(client, path):
    b64 = base64.b64encode(path.read_bytes()).decode()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": USER},
            ]},
        ],
        max_tokens=2048,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or ""
    ans = strip_think(raw).upper()
    m = re.search(r"\b(YES|NO)\b", ans)
    return m.group(1) if m else "?"


def load_cache():
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["id"]] = r["verdict"]
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300, help="number of pages to sample (0 = all)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    imgs = sorted(IMG_DIR.glob("*.png"))
    if args.samples and args.samples < len(imgs):
        random.seed(args.seed)
        imgs = random.sample(imgs, args.samples)

    # resume: skip pages already classified (only count clean YES/NO as done)
    cached = load_cache()
    results = {stem: v for stem, v in cached.items()}
    pending = [p for p in imgs if cached.get(p.stem) not in ("YES", "NO")]
    print(f"classifying {len(pending)} pages ({len(imgs)-len(pending)} cached) "
          f"with {len(KEYS)} keys x {args.workers} workers", flush=True)

    clients = [OpenAI(base_url=API_BASE, api_key=k) for k in KEYS]
    lock, done, cache_f = threading.Lock(), [0], open(CACHE, "a", encoding="utf-8")

    def work(i, path):
        for attempt in range(3):
            try:
                v = classify(clients[i % len(clients)], path)
                return path.stem, v
            except Exception as e:
                if attempt == 2:
                    return path.stem, f"ERR:{type(e).__name__}"
        return path.stem, "ERR"

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, i, p) for i, p in enumerate(pending)]
            for fut in as_completed(futs):
                stem, v = fut.result()
                with lock:
                    results[stem] = v
                    if v in ("YES", "NO"):  # only cache clean verdicts (retry errors next run)
                        cache_f.write(json.dumps({"id": stem, "verdict": v}) + "\n")
                        cache_f.flush()
                        os.fsync(cache_f.fileno())
                    done[0] += 1
                    if done[0] % 50 == 0:
                        yes = sum(1 for x in results.values() if x == "YES")
                        no = sum(1 for x in results.values() if x == "NO")
                        print(f"  [{done[0]}/{len(pending)}] YES={yes} NO={no}", flush=True)
    finally:
        cache_f.close()

    # aggregate over the requested image set only
    want = {p.stem for p in imgs}
    sub = {k: v for k, v in results.items() if k in want}
    yes = sum(1 for v in sub.values() if v == "YES")
    no = sum(1 for v in sub.values() if v == "NO")
    err = sum(1 for v in sub.values() if v.startswith("ERR") or v == "?")
    results = sub
    OUT.write_text(json.dumps({"total": len(results), "yes": yes, "no": no, "err": err,
                               "no_pages": sorted(k for k, v in results.items() if v == "NO"),
                               "results": results}, indent=2), encoding="utf-8")
    n = max(1, yes + no)
    print(f"\nDONE: {len(results)} pages | handwritten(YES)={yes} | printed-only(NO)={no} | errors={err}")
    print(f"PRINTED-ONLY FRACTION: {100*no/n:.1f}% of classifiable pages")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
