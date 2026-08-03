"""Unified live scorer for external (non-GLM) benchmark models.

As each model's generator writes per-page .raw/.tex/.time files, this re-derives
the GLM-OCR-style .tex from .raw, scores it against the references, compiles a
.pdf, writes a per-model cache, and merges a growing column into
benchmark_results.json — the same artifacts/schema the benchmark makes for
base/v3.1/v4/v4.1. ONE process = ONE writer, so multiple models never clobber
the shared results file.

Currently tracks: Baidu Unlimited-OCR (local GPU) + Mistral OCR 4 (API). A model
is only required for "completion" once its generator has actually started (its
*_gen_progress.json exists), so Mistral being absent never blocks Baidu.
"""
import os, sys, json, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DASH = Path(r"D:\ocr2tex\dashboard")
sys.path.insert(0, str(DASH))
from benchmark_glm_ocr import (score, summarize, compile_tex, cache_file,
                               load_cache, append_result, model_label, safe_label, DATASET_DIR)
from baidu_format import to_glm_format, ft_format

OUTPUT_BASE = Path(r"D:\ocr2tex\output")
CACHE_DIR = OUTPUT_BASE / "bench_cache"
RESULTS = OUTPUT_BASE / "benchmark_results.json"
LOG = OUTPUT_BASE / "score_models.log"

# fmt: "glm" -> base/markdown output needs GLM envelope + furniture strip;
#      "ft"  -> fine-tuned model already emits a complete LaTeX document (pass-through)
MODELS = [
    {"spec": "Baidu Unlimited-OCR", "progress": OUTPUT_BASE / "baidu_gen_progress.json", "fmt": "glm"},
    {"spec": "Baidu OCR FT",        "progress": OUTPUT_BASE / "baidu_ft_gen_progress.json", "fmt": "ft"},
    {"spec": "Mistral OCR 4",       "progress": OUTPUT_BASE / "mistral_gen_progress.json", "fmt": "glm"},
]
for m in MODELS:
    m["safe"] = safe_label(m["spec"])
    m["tex_dir"] = OUTPUT_BASE / "bench_outputs" / m["safe"]
    m["cache"] = cache_file(CACHE_DIR, m["spec"])


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_samples():
    rows = [json.loads(l) for l in (DATASET_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    refs = {s["id"]: next((c["content"] for c in s["conversations"] if c["role"] == "assistant"), "") for s in rows}
    return [s["id"] for s in rows], refs


def page_ready(tex_dir, pid):
    return (tex_dir / f"{pid}.time").exists()  # written last by the generator


def score_one(m, pid, ref):
    rawf = m["tex_dir"] / f"{pid}.raw"
    raw = rawf.read_text(encoding="utf-8", errors="replace") if rawf.exists() else ""
    hyp = (ft_format if m.get("fmt") == "ft" else to_glm_format)(raw)
    (m["tex_dir"] / f"{pid}.tex").write_text(hyp, encoding="utf-8")
    row = score(ref, hyp)
    row["id"] = pid
    tf = m["tex_dir"] / f"{pid}.time"
    try:
        row["latency_s"] = float(tf.read_text().strip()) if tf.exists() else 0.0
    except Exception:
        row["latency_s"] = 0.0
    row["hyp"] = hyp
    ok, _ = compile_tex(hyp, m["tex_dir"] / f"{pid}.pdf")
    row["compile_ok"] = bool(ok)
    return row


def gen_status(m):
    try:
        return json.loads(m["progress"].read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def merge_all(sample_ids, caches, running):
    """Single-writer merge of every model's column into benchmark_results.json."""
    if not RESULTS.exists():
        return
    try:
        res = json.loads(RESULTS.read_text(encoding="utf-8"))
    except Exception:
        return
    keys = res.setdefault("model_keys", [])
    models = res.setdefault("models", {})
    per = res.setdefault("per_sample", {})
    for m in MODELS:
        done = caches.get(m["spec"], {})
        if not done:
            continue
        rows = [done[sid] for sid in sample_ids if sid in done]
        summary = summarize(rows)
        bkey = next((k for k in keys if models.get(k, {}).get("spec") == m["spec"]), None)
        if bkey is None:
            bkey = chr(ord("a") + len(keys))
            keys.append(bkey)
        models[bkey] = {"label": model_label(m["spec"]), "spec": m["spec"], "summary": summary}
        per[bkey] = rows
    res["state"] = "running" if running else "done"
    if running:
        # mark the actively-generating model so the dashboard shows a live "page N/700"
        # + the running (blue) indicator instead of a stale/yellow bar
        act = next((m for m in MODELS if gen_status(m) == "running"), None)
        if act:
            ak = next((k for k in keys if models.get(k, {}).get("spec") == act["spec"]), None)
            scored = len([sid for sid in sample_ids if sid in caches.get(act["spec"], {})])
            res["progress"] = {"model": model_label(act["spec"]), "key": ak,
                               "done": scored, "total": len(sample_ids)}
    else:
        res.pop("progress", None)
    try:  # best CER reduction vs base, across all columns
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
    for _ in range(20):  # Windows: retry if dashboard holds the file open mid-read
        try:
            os.replace(tmp, RESULTS); return
        except PermissionError:
            time.sleep(0.25)
    log("WARN: results file locked; retry next pass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=15)
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sample_ids, refs = load_samples()
    total = len(sample_ids)
    log(f"unified scorer up: {total} pages, tracking {[m['spec'] for m in MODELS]}")

    pool = ThreadPoolExecutor(max_workers=4)
    while True:
      try:   # never let a transient file race / json error kill the long-lived scorer
        caches, any_pending = {}, False
        for m in MODELS:
            done = load_cache(m["cache"])
            pending = [sid for sid in sample_ids
                       if page_ready(m["tex_dir"], sid) and (sid not in done or "compile_ok" not in done[sid])]
            if pending:
                any_pending = True
                futs = {sid: pool.submit(score_one, m, sid, refs[sid]) for sid in pending}
                for sid, fut in futs.items():
                    try:
                        row = fut.result()
                    except Exception as e:
                        log(f"  {m['safe']} score failed {sid}: {type(e).__name__}: {str(e)[:80]}")
                        continue
                    append_result(m["cache"], row)
                    done[sid] = row
                sm = summarize([done[s] for s in sample_ids if s in done])
                log(f"{m['safe']}: scored {len([s for s in sample_ids if s in done])}/{total} "
                    f"| mean_cer={sm.get('mean_cer')} ncer={sm.get('norm_cer')} compile={sm.get('compile_rate')}%")
            caches[m["spec"]] = done

        # a model is 'required' only once its generator has started, and 'done' when
        # its generator finished AND every page it actually produced is scored (so a
        # partial run — e.g. base Baidu stopped at 158/700 — can still complete)
        def model_done(m):
            if gen_status(m) != "done":
                return False
            cache = caches.get(m["spec"], {})
            return all(sid in cache for sid in sample_ids if page_ready(m["tex_dir"], sid))
        required = [m for m in MODELS if m["progress"].exists()]
        complete = bool(required) and all(model_done(m) for m in required)
        if any_pending or complete:
            merge_all(sample_ids, caches, running=not complete)
        if complete:
            log("ALL TRACKED MODELS COMPLETE")
            break
      except Exception as e:
        log(f"loop error (continuing): {type(e).__name__}: {str(e)[:120]}")
      time.sleep(args.poll)
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
