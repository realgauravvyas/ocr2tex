"""Live scorer: as gen_baidu_for_benchmark.py writes Unlimited-OCR .tex files,
score them against the GLM-OCR references, compile a .pdf, append to a per-model
cache, and merge a growing "Baidu Unlimited-OCR" column into benchmark_results.json
— byte-for-byte the same artifacts/schema the benchmark makes for base/v3.1/v4/v4.1.

Runs in the dashboard's Python env (has numpy + pdflatex). CPU only — never
touches the GPU, so it runs safely alongside the generation job.
"""
import os, sys, json, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DASH = Path(r"D:\ocr2tex\dashboard")
sys.path.insert(0, str(DASH))
from benchmark_glm_ocr import (score, summarize, compile_tex, cache_file,
                               load_cache, append_result, model_label, safe_label, DATASET_DIR)
from baidu_format import to_glm_format

OUTPUT_BASE = Path(r"D:\ocr2tex\output")
SPEC = "Baidu Unlimited-OCR"
SAFE = safe_label(SPEC)
TEX_DIR = OUTPUT_BASE / "bench_outputs" / SAFE
CACHE_DIR = OUTPUT_BASE / "bench_cache"
RESULTS = OUTPUT_BASE / "benchmark_results.json"
GEN_PROGRESS = OUTPUT_BASE / "baidu_gen_progress.json"
LOG = OUTPUT_BASE / "baidu_score.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_samples():
    rows = []
    for line in (DATASET_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    refs = {s["id"]: next((c["content"] for c in s["conversations"] if c["role"] == "assistant"), "")
            for s in rows}
    return [s["id"] for s in rows], refs


def page_ready(pid):
    # .time is written last by the generator, so it marks a fully-written page
    return (TEX_DIR / f"{pid}.time").exists()


def score_one(pid, ref, cache_path):
    """Re-derive the GLM-style .tex from the raw det-stream (single source of truth),
    then score + compile one page; returns the cache row (same schema as benchmark)."""
    rawf = TEX_DIR / f"{pid}.raw"
    raw = rawf.read_text(encoding="utf-8", errors="replace") if rawf.exists() else ""
    hyp = to_glm_format(raw)
    (TEX_DIR / f"{pid}.tex").write_text(hyp, encoding="utf-8")   # canonical .tex shown in Inspector
    row = score(ref, hyp)
    row["id"] = pid
    tf = TEX_DIR / f"{pid}.time"
    try:
        row["latency_s"] = float(tf.read_text().strip()) if tf.exists() else 0.0
    except Exception:
        row["latency_s"] = 0.0
    row["hyp"] = hyp
    ok, _ = compile_tex(hyp, TEX_DIR / f"{pid}.pdf")   # makes the .pdf like other models
    row["compile_ok"] = bool(ok)
    return row


def merge_into_results(sample_ids, done, summary, running, scored, total):
    """Add/refresh the Baidu column in benchmark_results.json (atomic, preserves a-d).
    While scoring, set state=running + progress so the EXISTING benchmark panel shows
    the Baidu bar filling live; set state=done when finished."""
    if not RESULTS.exists():
        return
    try:
        res = json.loads(RESULTS.read_text(encoding="utf-8"))
    except Exception:
        return
    keys = res.setdefault("model_keys", [])
    models = res.setdefault("models", {})
    per = res.setdefault("per_sample", {})
    bkey = next((k for k in keys if models.get(k, {}).get("spec") == SPEC), None)
    if bkey is None:
        bkey = chr(ord("a") + len(keys))
        keys.append(bkey)
    models[bkey] = {"label": model_label(SPEC), "spec": SPEC, "summary": summary}
    per[bkey] = [done[sid] for sid in sample_ids if sid in done]
    res["state"] = "running" if running else "done"
    if running:
        res["progress"] = {"model": model_label(SPEC), "key": bkey, "done": scored, "total": total}
    else:
        res.pop("progress", None)
    # recompute "best CER reduction vs base" across all columns incl. Baidu
    try:
        base_cer = models[keys[0]]["summary"].get("mean_cer")
        valid = {k: models[k]["summary"].get("mean_cer") for k in keys
                 if models[k].get("summary", {}).get("mean_cer") is not None}
        if base_cer and valid:
            bk = min(valid, key=valid.get)
            res["improvement"] = {"best_key": bk, "best_label": models[bk]["label"],
                                  "ref_label": models[keys[0]]["label"],
                                  "cer_reduction_pct": round(100.0 * (base_cer - valid[bk]) / max(1e-9, base_cer), 1)}
    except Exception:
        pass
    tmp = RESULTS.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    # On Windows os.replace fails with Access-Denied if the dashboard is mid-read,
    # so retry briefly rather than crash the scorer.
    for attempt in range(20):
        try:
            os.replace(tmp, RESULTS)
            return
        except PermissionError:
            time.sleep(0.25)
    log("WARN: could not replace results file after retries (locked); will retry next pass")


def gen_done():
    try:
        return json.loads(GEN_PROGRESS.read_text(encoding="utf-8")).get("status") == "done"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=15, help="seconds between scan passes")
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    args = ap.parse_args()
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = cache_file(CACHE_DIR, SPEC)
    sample_ids, refs = load_samples()
    total = len(sample_ids)
    log(f"scorer up: {total} test pages, cache={cache_path.name}")

    pool = ThreadPoolExecutor(max_workers=4)
    while True:
        done = load_cache(cache_path)
        pending = [sid for sid in sample_ids
                   if page_ready(sid) and (sid not in done or "compile_ok" not in done[sid])]
        if pending:
            futs = {sid: pool.submit(score_one, sid, refs[sid], cache_path) for sid in pending}
            for sid, fut in futs.items():
                try:
                    row = fut.result()
                except Exception as e:
                    log(f"  score failed {sid}: {type(e).__name__}: {str(e)[:80]}")
                    continue
                append_result(cache_path, row)
                done[sid] = row
        scored = len([s for s in sample_ids if s in done])
        complete = (gen_done() and scored >= total)
        if pending or complete:
            summary = summarize([done[s] for s in sample_ids if s in done])
            merge_into_results(sample_ids, done, summary, running=not complete, scored=scored, total=total)
            if pending:
                log(f"scored {scored}/{total} | mean_cer={summary.get('mean_cer')} "
                    f"ncer={summary.get('norm_cer')} compile={summary.get('compile_rate')}%")
        if args.once or complete:
            log(f"DONE scoring {scored}/{total}")
            break
        time.sleep(args.poll)
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
