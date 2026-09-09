r"""Cheap validation of the v3 decode fixes on the EXISTING v1 adapter.

Run this BEFORE retraining. It answers, in ~20 minutes of GPU instead of ~40
hours, whether the two zero-training fixes actually recover the gap:

  A  legacy      ring KV cache ON, total max_length=2048, no_repeat 35/128,
                 legacy ft_format  -> should reproduce the recorded v1 column
  B  ring off    ring KV cache OFF, max_new_tokens, no_repeat 35/128, v3 format
  C  ring off,   ring OFF, max_new_tokens, NO ngram processor, v3 format
     no ngram
  D  fmt only    exactly A's decode, but scored through the v3 repairing
                 formatter -> isolates how much is post-processing alone

Pages are sampled half from the set that truncated under v1 (read-only lookup in
D:\ocr2tex\output\bench_outputs\Baidu_OCR_FT) and half at random, so the table
shows both "does it fix the broken pages" and "does it hurt the good ones".

Writes probe/probe_results.json + a printed table INSIDE this folder. Reads the
dataset, the v1 adapter and the v1 outputs read-only; writes nothing elsewhere.
"""
import os, sys, json, time, random, argparse
from pathlib import Path

V3 = Path(__file__).resolve().parent
sys.path.insert(0, str(V3))

import torch
from common_v3 import (load_base, helpers, build_inputs, generate_latex, causal_lm,
                       read_jsonl, target_of, DATA_DIR, IMAGE_DIR, write_json_atomic)
from format_v3 import ft_format_v3
import metrics_v3

V1_ADAPTER = Path(r"D:\Claude Code\BaiduOCR\finetune\baidu-ocr-math-v1")   # READ-ONLY
V1_OUTPUTS = Path(r"D:\ocr2tex\output\bench_outputs\Baidu_OCR_FT")          # READ-ONLY
OUT = V3 / "probe"
PROMPT = "<image>Convert the handwriting to a complete LaTeX document."      # must match training

VARIANTS = {
    "A_legacy":      dict(ring=True,  budget="total2048", ngram=35, repair=False),
    "B_ringoff":     dict(ring=False, budget="new",       ngram=35, repair=True),
    "C_ringoff_nong": dict(ring=False, budget="new",      ngram=0,  repair=True),
    "D_fmtonly":     dict(ring=True,  budget="total2048", ngram=35, repair=True),
}


def pick_adapter(root: Path):
    for c in (root / "best", root / "final", root):
        if (c / "adapter_config.json").exists():
            return c
    return None


def truncated_v1_ids():
    """Page ids whose v1 generation never emitted \\end{document}."""
    if not V1_OUTPUTS.exists():
        return set()
    out = set()
    for p in V1_OUTPUTS.glob("*.raw"):
        try:
            if "\\end{document}" not in p.read_text(encoding="utf-8", errors="replace"):
                out.add(p.stem)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=24)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--variants", default="A_legacy,B_ringoff,C_ringoff_nong,D_fmtonly")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--page-timeout", type=float, default=180.0)
    ap.add_argument("--max-crops", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compile", action="store_true", help="skip pdflatex (faster)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"metrics: {metrics_v3.reason()}", flush=True)

    rows = read_jsonl(DATA_DIR / "test.jsonl")
    refs = {r["id"]: target_of(r) for r in rows}
    bad = truncated_v1_ids()
    good = [r["id"] for r in rows if r["id"] not in bad]
    badl = [r["id"] for r in rows if r["id"] in bad]
    random.seed(args.seed)
    random.shuffle(good)
    random.shuffle(badl)
    half = args.pages // 2
    pages = badl[:half] + good[: args.pages - min(half, len(badl))]
    print(f"probing {len(pages)} pages ({min(half, len(badl))} previously truncated, "
          f"{len(pages) - min(half, len(badl))} previously clean)", flush=True)

    adapter = Path(args.adapter) if args.adapter else pick_adapter(V1_ADAPTER)
    if adapter is None:
        sys.exit(f"ERROR: no adapter under {V1_ADAPTER}")
    print(f"adapter: {adapter}", flush=True)

    print("loading base + adapter...", flush=True)
    t0 = time.time()
    model, tok = load_base()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().eval()
    if dev == "cuda":
        model = model.cuda()
    M = helpers(model)
    print(f"ready in {time.time() - t0:.0f}s", flush=True)

    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = {v: [] for v in want}
    raw_dump = {v: {} for v in want}

    for i, pid in enumerate(pages):
        ex = build_inputs(M, tok, IMAGE_DIR / f"{pid}.png", PROMPT, max_crops=args.max_crops)
        plen = ex["prompt_len"]
        for v in want:
            cfg = VARIANTS[v]
            budget = (2048 - plen) if cfg["budget"] == "total2048" else args.max_new_tokens
            budget = max(16, budget)
            t1 = time.time()
            raw = generate_latex(model, tok, ex, dev, M=M, max_new_tokens=budget,
                                 ring=cfg["ring"], no_repeat_ngram_size=cfg["ngram"],
                                 ngram_window=128 if cfg["ngram"] else 0,
                                 max_time=args.page_timeout)
            dt = time.time() - t1
            hyp = ft_format_v3(raw, repair=cfg["repair"])
            row = metrics_v3.score(refs[pid], hyp)
            row["id"] = pid
            row["latency_s"] = round(dt, 2)
            row["raw_complete"] = "\\end{document}" in raw
            row["was_truncated_in_v1"] = pid in bad
            if not args.no_compile:
                ok, _ = metrics_v3.compile_tex(hyp)
                row["compile_ok"] = None if ok is None else bool(ok)
            results[v].append(row)
            raw_dump[v][pid] = raw
        done = [f"{v}:cer={results[v][-1]['cer']:.3f}{'' if results[v][-1]['raw_complete'] else '!'}"
                for v in want]
        print(f"  [{i+1}/{len(pages)}] {pid} plen={plen} " + " ".join(done), flush=True)
        write_json_atomic(OUT / "probe_results.json",
                          {v: metrics_v3.summarize(results[v]) for v in want})

    print("\n" + "=" * 96)
    hdr = f"{'variant':<18}{'CER':>8}{'nCER':>8}{'struct%':>9}{'compile%':>10}{'complete%':>11}{'len':>7}{'sec':>7}"
    print(hdr)
    print("-" * 96)
    summary = {}
    for v in want:
        rs = results[v]
        s = metrics_v3.summarize(rs)
        comp = [r["compile_ok"] for r in rs if r.get("compile_ok") is not None]
        s["complete_pct"] = round(100.0 * sum(r["raw_complete"] for r in rs) / len(rs), 1)
        s["compile_rate"] = round(100.0 * sum(comp) / len(comp), 1) if comp else None
        s["mean_latency_s"] = round(sum(r["latency_s"] for r in rs) / len(rs), 1)
        summary[v] = s
        print(f"{v:<18}{s.get('mean_cer', 0):>8.4f}{s.get('norm_cer', s.get('ncer', 0)) or 0:>8.4f}"
              f"{s.get('struct_pct', 0):>9.1f}"
              f"{(s['compile_rate'] if s['compile_rate'] is not None else -1):>10.1f}"
              f"{s['complete_pct']:>11.1f}{s.get('len_rate', 0):>7.2f}{s['mean_latency_s']:>7.1f}")
    print("=" * 96)
    print("subset of pages that truncated under v1:")
    for v in want:
        rs = [r for r in results[v] if r["was_truncated_in_v1"]]
        if rs:
            print(f"  {v:<18} CER={sum(r['cer'] for r in rs)/len(rs):.4f}  "
                  f"complete={100.0*sum(r['raw_complete'] for r in rs)/len(rs):.0f}%")

    write_json_atomic(OUT / "probe_results.json",
                      {"summary": summary, "per_page": results, "pages": pages,
                       "adapter": str(adapter), "metrics": metrics_v3.reason()})
    (OUT / "raw_samples.json").write_text(json.dumps(raw_dump)[:20_000_000], encoding="utf-8")
    print(f"\nwrote {OUT/'probe_results.json'}")


if __name__ == "__main__":
    main()
