r"""Run a v3 (or v1/v2) adapter over the 700-page test set with the fixed decode.

Writes ONLY under D:\Claude Code\BaiduOCR-v3\bench\<label>\ -- it never touches
D:\ocr2tex\output\bench_outputs or benchmark_results.json, so the recorded v1/v2
columns and the dashboard stay exactly as they are.

Decode differences vs gen_baidu_ft_for_benchmark.py
---------------------------------------------------
* ring KV cache OFF (--ring to re-enable for an A/B). At decode time the
  checkpoint arms a 128-slot ring buffer, so a generated token can only attend
  to the prompt plus the last 128 tokens it produced, on recycled slots that
  keep their original RoPE phase. Training has no cache and therefore full
  attention. Targets average 532 tokens; the mismatch is what produces the
  "$\in I_{36}$ \quad $\in I_{37}$ ..." loops seen on 103 of the 124 v1 pages
  that never reached \end{document}.
* --max-new-tokens, not a total max_length. The old 2048 total cap was shared
  with an image prompt of ~903 tokens (2x3 crops) and up to ~1900 on a tall
  page, so the answer budget silently depended on page geometry.
* infer() is bypassed: same inputs, but no bbox drawing / matplotlib / temp-dir
  round trip, and the special-token cleanup happens in format_v3 instead.
* max_num=32 crops, matching dynamic_preprocess's own default (v1 trained at 16
  and decoded at 32).
"""
import os, sys, json, time, argparse
from pathlib import Path

V3 = Path(__file__).resolve().parent
sys.path.insert(0, str(V3))

import torch
from common_v3 import (load_base, helpers, build_inputs, generate_latex,
                       read_jsonl, DATA_DIR, IMAGE_DIR, write_json_atomic)
from format_v3 import ft_format_v3

DEFAULT_ADAPTER = V3 / "out" / "baidu-ocr-math-v3"
BENCH = V3 / "bench"
PROMPT = "<image>Convert the handwriting to a complete LaTeX document."   # must match training


def pick_adapter(root: Path):
    for c in (root / "best_cer", root / "best", root / "final", root / "best_loss", root):
        if (c / "adapter_config.json").exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="Baidu_OCR_FT_v3", help="subfolder of bench/")
    ap.add_argument("--adapter", default="", help="blank -> best_cer/ else final/ under out/")
    ap.add_argument("--samples", type=int, default=700)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-crops", type=int, default=32)
    ap.add_argument("--page-timeout", type=float, default=180.0)
    ap.add_argument("--ring", action="store_true", help="re-arm the 128-slot ring KV cache (A/B only)")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--ngram-window", type=int, default=128)
    ap.add_argument("--redo", action="store_true", help="regenerate pages that already have output")
    ap.add_argument("--page-ids", default="", help="comma-separated page ids -- run only these, ignores --samples")
    args = ap.parse_args()

    out_dir = BENCH / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = BENCH / f"{args.label}_progress.json"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    adapter = Path(args.adapter) if args.adapter else pick_adapter(DEFAULT_ADAPTER)
    if adapter is None or not (adapter / "adapter_config.json").exists():
        write_json_atomic(progress, {"status": "error", "error": "adapter not found"})
        sys.exit(f"ERROR: no LoRA adapter found (looked under {DEFAULT_ADAPTER})")
    print(f"adapter: {adapter}", flush=True)

    rows = read_jsonl(DATA_DIR / "test.jsonl")
    if args.page_ids:
        want = set(x.strip() for x in args.page_ids.split(",") if x.strip())
        rows = [r for r in rows if r["id"] in want]
        missing = want - {r["id"] for r in rows}
        if missing:
            sys.exit(f"ERROR: page ids not found in test.jsonl: {sorted(missing)}")
    else:
        rows = rows[: args.samples]

    def done(r):
        f = out_dir / f"{r['id']}.raw"
        return f.exists() and (out_dir / f"{r['id']}.time").exists()

    pending = rows if args.redo else [r for r in rows if not done(r)]
    total, existing = len(rows), len(rows) - len(pending)
    print(f"{args.label}: {total} target, {len(pending)} to run "
          f"| ring={'ON' if args.ring else 'OFF'} max_new_tokens={args.max_new_tokens}", flush=True)
    write_json_atomic(progress, {"status": "loading", "model": args.label, "total": total,
                                 "done": existing, "adapter": str(adapter),
                                 "ring": args.ring, "started_at": time.time()})

    print("loading base + adapter...", flush=True)
    t0 = time.time()
    model, tok = load_base()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().eval()
    if dev == "cuda":
        model = model.cuda()
    M = helpers(model)
    print(f"model ready in {time.time()-t0:.0f}s", flush=True)

    times, truncated = [], 0
    for i, r in enumerate(pending):
        pid = r["id"]
        t1 = time.time()
        try:
            ex = build_inputs(M, tok, IMAGE_DIR / f"{pid}.png", PROMPT, max_crops=args.max_crops)
            raw = generate_latex(model, tok, ex, dev, M=M,
                                 max_new_tokens=args.max_new_tokens, ring=args.ring,
                                 no_repeat_ngram_size=args.no_repeat_ngram_size,
                                 ngram_window=args.ngram_window,
                                 max_time=args.page_timeout)
        except Exception as e:
            print(f"  {pid} FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
            raw = ""
            torch.cuda.empty_cache()
        dt = time.time() - t1
        tex = ft_format_v3(raw)
        complete = "\\end{document}" in raw
        truncated += (not complete)
        times.append(dt)

        (out_dir / f"{pid}.raw").write_text(raw, encoding="utf-8")
        (out_dir / f"{pid}.tex").write_text(tex, encoding="utf-8")
        (out_dir / f"{pid}.time").write_text(str(round(dt, 2)), encoding="utf-8")

        n = existing + i + 1
        eta = (sum(times) / len(times)) * (len(pending) - i - 1) / 3600
        print(f"  [{n}/{total}] {pid}: {len(tex)} chars, {dt:.1f}s"
              f"{'' if complete else '  TRUNCATED'}, ETA {eta:.2f}h", flush=True)
        write_json_atomic(progress, {"status": "running", "model": args.label, "done": n,
                                     "total": total, "current": pid,
                                     "pct": round(100 * n / total, 1), "eta_h": round(eta, 2),
                                     "avg_s": round(sum(times) / len(times), 1),
                                     "truncated": truncated, "ring": args.ring,
                                     "adapter": str(adapter)})

    write_json_atomic(progress, {"status": "done", "model": args.label, "done": total,
                                 "total": total, "truncated": truncated, "ring": args.ring,
                                 "adapter": str(adapter), "finished_at": time.time()})
    print(f"DONE  truncated={truncated}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
