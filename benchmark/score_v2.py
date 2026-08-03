"""EXPERIMENT: score the Baidu FT v2 column and write a COPY of the benchmark
results to this experiment folder. The canonical D:\\ocr2tex\\output\\
benchmark_results.json is NEVER modified, so your live dashboard is untouched.

Reads:
  bench_pages : bench_outputs/Baidu_OCR_FT_v2/<id>.raw|.time  (from gen_baidu_ft_v2)
  references  : workspace/9_split/test.jsonl
Writes:
  experiments/baidu_v2/benchmark_results_with_v2.json   (existing columns + Baidu OCR FT v2)
"""

import os, sys, json, time
from pathlib import Path

sys.path.insert(0, r"D:\ocr2tex\dashboard")
from benchmark_glm_ocr import score, summarize, compile_tex
from baidu_format import ft_format

OUTPUT_BASE = Path(r"D:\ocr2tex\output")
RESULTS_SRC = OUTPUT_BASE / "benchmark_results.json"
TEST = Path(r"D:\ocr2tex\workspace\9_split\test.jsonl")
OUT_DIR = Path(__file__).parent
RESULTS_DST = OUT_DIR / "benchmark_results_with_v2.json"

TEX_DIR = OUTPUT_BASE / "bench_outputs" / "Baidu_OCR_FT_v2"
PROGRESS = OUTPUT_BASE / "baidu_ft_v2_gen_progress.json"


def load_refs():
    rows = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    # id → reference latex, and id → image basename (e.g. "foo" for images/foo.png)
    refs = {}
    for s in rows:
        asn = next(c["content"] for c in s["conversations"] if c["role"] == "assistant")
        img_base = Path(s["image"]).stem          # strip "images/" and ".png"
        refs[s["id"]] = {"ref": asn, "img": img_base}
    return refs


def main():
    refs = load_refs()
    done = {}
    for t in TEX_DIR.glob("*.time"):
        pid = t.stem
        if pid not in refs:
            continue
        rawf = TEX_DIR / f"{pid}.raw"
        raw = rawf.read_text(encoding="utf-8", errors="replace") if rawf.exists() else ""
        hyp = ft_format(raw)
        row = score(refs[pid]["ref"], hyp)
        row["id"] = pid
        try:
            row["latency_s"] = float(t.read_text().strip())
        except Exception:
            row["latency_s"] = 0.0
        row["hyp"] = hyp
        ok, _ = compile_tex(hyp, TEX_DIR / f"{pid}.pdf")
        row["compile_ok"] = bool(ok)
        done[pid] = row

    sample_ids = list(refs.keys())
    rows = [done[sid] for sid in sample_ids if sid in done]
    summary = summarize(rows)

    res = json.loads(RESULTS_SRC.read_text(encoding="utf-8"))
    keys = res.setdefault("model_keys", [])
    models = res.setdefault("models", {})
    per = res.setdefault("per_sample", {})
    # drop any previous v2 entry so re-runs are idempotent
    for k in [k for k in keys if models.get(k, {}).get("spec") == "Baidu OCR FT v2"]:
        keys.remove(k); models.pop(k, None); per.pop(k, None)
    # find the first free letter key (a, b, c, ...)
    for i in range(26):
        cand = chr(ord("a") + i)
        if cand not in models:
            bkey = cand
            break
    keys.append(bkey)
    models[bkey] = {"label": "Baidu OCR FT v2", "spec": "Baidu OCR FT v2", "summary": summary}
    per[bkey] = rows

    RESULTS_DST.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"scored {len(rows)}/{len(sample_ids)} pages -> {RESULTS_DST}", flush=True)
    print(f"Baidu OCR FT v2: cer={summary.get('mean_cer')} ncer={summary.get('norm_cer')} "
          f"bleu={summary.get('bleu')} compile={summary.get('compile_rate')}%", flush=True)


if __name__ == "__main__":
    main()
