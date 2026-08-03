"""EXPERIMENT: score the Baidu FT v2-BEST column (the `best/` checkpoint adapter,
val 0.2096) and append it to the experiment COPY of the benchmark results.
Never touches the canonical benchmark_results.json.

Reads:   bench_outputs/Baidu_OCR_FT_v2_best/<id>.raw|.time
Writes:  experiments/baidu_v2/benchmark_results_with_v2.json  (adds v2-best column)
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
TEX_DIR = OUTPUT_BASE / "bench_outputs" / "Baidu_OCR_FT_v2_best"
LABEL = "Baidu OCR FT v2 (best ckpt)"


def load_refs():
    rows = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    refs = {}
    for s in rows:
        asn = next(c["content"] for c in s["conversations"] if c["role"] == "assistant")
        refs[s["id"]] = {"ref": asn, "img": Path(s["image"]).stem}
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
    if len(rows) < 600:
        print(f"ABORT: only {len(rows)}/700 pages present - generation not finished yet", flush=True)
        sys.exit(3)
    summary = summarize(rows)

    res = json.loads(RESULTS_SRC.read_text(encoding="utf-8-sig"))
    keys = res.setdefault("model_keys", [])
    models = res.setdefault("models", {})
    per = res.setdefault("per_sample", {})
    for k in [k for k in keys if models.get(k, {}).get("spec") == LABEL]:
        keys.remove(k); models.pop(k, None); per.pop(k, None)
    for i in range(26):
        cand = chr(ord("a") + i)
        if cand not in models:
            bkey = cand
            break
    keys.append(bkey)
    models[bkey] = {"label": LABEL, "spec": LABEL, "summary": summary}
    per[bkey] = rows

    RESULTS_DST.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"scored {len(rows)}/700 pages -> {RESULTS_DST}", flush=True)
    print(f"{LABEL}: cer={summary.get('mean_cer')} ncer={summary.get('norm_cer')} "
          f"bleu={summary.get('bleu')} compile={summary.get('compile_rate')}%", flush=True)


if __name__ == "__main__":
    main()
