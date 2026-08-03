"""Self-consistency verification for salvaged (temp>0.1) annotations.

Every page annotated during the salvage pass gets ONE additional independent
annotation at the same temperature. If the two independent samples agree
(normalized character error rate <= --threshold), the annotation is considered
anchored to the page content and kept. If they disagree, or the verification
call fails, the page is marked REJECT (conservative: unverifiable => excluded).

Writes workspace/verify_report.json: {"keep": [...], "reject": [...], "details": {...}}
Does NOT delete anything itself - exclusion is applied by the caller.
"""

import json
import time
import base64
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

WORK = Path(r"D:\ocr2tex\workspace")
ANN = WORK / "6_annotations"
IMG = WORK / "5_prepared"

API_BASE = "https://api.tokenrouter.com/v1"
MODEL = "MiniMax-M3"


def normalize(text):
    return " ".join(text.split())


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    a_arr = np.frombuffer(a.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    b_arr = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    n = len(b_arr)
    idx = np.arange(n + 1, dtype=np.int64)
    prev = idx.copy()
    for i, ca in enumerate(a_arr):
        cand = np.empty(n + 1, dtype=np.int64)
        cand[0] = i + 1
        cand[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b_arr != ca))
        running = np.minimum.accumulate(cand - idx)
        prev = np.minimum(cand, running + idx)
    return int(prev[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=str, required=True, help="comma-separated API keys")
    ap.add_argument("--temperature", type=float, default=0.45)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=16384)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from app import ANNOTATION_SYSTEM_PROMPT, ANNOTATION_USER_PROMPT, clean_latex_response, is_valid_latex
    from openai import OpenAI

    meta = json.loads((WORK / "salvaged_pages.json").read_text(encoding="utf-8"))
    cutoff = meta["cutoff"]
    salvaged = sorted(f.stem for f in ANN.glob("*.tex") if f.stat().st_mtime >= cutoff - 5)
    print(f"verifying {len(salvaged)} salvaged pages (cutoff {cutoff})", flush=True)

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    clients = [OpenAI(base_url=API_BASE, api_key=k) for k in keys]

    lock = threading.Lock()
    results = {}

    def verify_one(i, stem):
        client = clients[i % len(clients)]
        original = normalize((ANN / f"{stem}.tex").read_text(encoding="utf-8"))
        img_path = IMG / f"{stem}.png"
        for attempt in range(2):
            try:
                b64 = base64.b64encode(img_path.read_bytes()).decode()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": ANNOTATION_USER_PROMPT},
                        ]},
                    ],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                raw = resp.choices[0].message.content or ""
                second = clean_latex_response(raw)
                if not is_valid_latex(second):
                    continue
                second_n = normalize(second)
                dist = levenshtein(original, second_n)
                cer = dist / max(1, len(original))
                return stem, ("keep" if cer <= args.threshold else "reject"), round(cer, 4)
            except Exception as e:
                if attempt == 1:
                    return stem, "reject", None
                time.sleep(3)
        return stem, "reject", None

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(verify_one, i, s) for i, s in enumerate(salvaged)]
        for fut in as_completed(futures):
            stem, verdict, cer = fut.result()
            with lock:
                results[stem] = {"verdict": verdict, "cer": cer}
                done += 1
                if done % 50 == 0:
                    kept = sum(1 for r in results.values() if r["verdict"] == "keep")
                    print(f"[{done}/{len(salvaged)}] keep={kept} reject={done-kept}", flush=True)

    keep = sorted(s for s, r in results.items() if r["verdict"] == "keep")
    reject = sorted(s for s, r in results.items() if r["verdict"] == "reject")
    report = {
        "checked": len(salvaged),
        "keep": keep,
        "reject": reject,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "details": results,
    }
    (WORK / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cers = [r["cer"] for r in results.values() if r["cer"] is not None]
    print(f"VERIFY DONE: {len(keep)} keep, {len(reject)} reject "
          f"(median agreement CER: {np.median(cers):.4f})", flush=True)


if __name__ == "__main__":
    main()
