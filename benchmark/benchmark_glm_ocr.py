"""Multi-model benchmark for handwritten-math -> LaTeX (GLM-OCR + adapters).

Compares 2-3 models (base GLM-OCR and/or any LoRA adapters) on held-out test
pages. Each model generates LaTeX which is scored against the MiniMax M3
reference annotation with a full metric suite:

  CER       character error rate (edit dist / ref chars)        lower better
  CharSim   1 - CER, clamped to [0,1]                            higher better
  TokEdit   token-level error rate (LaTeX-aware tokenization)    lower better
  BLEU      BLEU-4, sentence-level, epsilon-smoothed             higher better
  ROUGE-L   LCS-based F1 over tokens                             higher better
  Math-F1   bag-of-tokens F1 (symbol presence, order-free)       higher better
  chrF      character n-gram F-score (robust to tokenization)    higher better
  Norm-CER  CER after canonicalizing LaTeX (style-robust)        lower better
  Struct    % of outputs that are complete, balanced LaTeX docs  higher better
  Compile   % of outputs that compile with pdflatex (non-base)   higher better
  LenRate   mean hyp/ref length ratio (1.0 = same length)        ~1.0 best
  Latency   mean generation seconds                              lower better

Artifacts saved under <output_dir>/bench_outputs/:
  <model_label>/<page_id>.tex   every model's generated LaTeX (always)
  <model_label>/<page_id>.pdf   compiled PDF (fine-tuned models only; base skipped)

The generated text is cached, so new metrics can be added later without re-running
the GPU. All text metrics use silver-standard references — treat them as RELATIVE
comparisons between models, not absolute accuracy. (A faithful render-based CDM
needs a character-detection pipeline; Norm-CER is the reliable style-robust proxy.)

Writes incremental progress + final summary to --output (JSON) for the dashboard.
"""

import os
import re
import json
import time
import hashlib
import argparse
import warnings
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
# reduce CUDA fragmentation OOM on a long multi-model run (must be set before CUDA init)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_NAME = "zai-org/GLM-OCR"
DATASET_DIR = Path(r"D:\ocr2tex\workspace\9_split")
MAX_IMAGE_TOKENS = 1536
MAX_TOK = 1200  # cap token sequences for O(n*m) metrics (LCS); generation dominates cost anyway

USER_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into a complete, compilable LaTeX document. Ignore printed text, "
    "student info, page numbers, cancelled work and rough work. Output only LaTeX."
)

TOKEN_RE = re.compile(r"\\[a-zA-Z]+\*?|\\.|[{}()\[\]^_&+=\-*/<>|]|[a-zA-Z0-9]+|\S")

# Path set in main(); when set, log_line() mirrors stdout into this file so the
# dashboard can tail it no matter how the benchmark was launched.
LOG_FILE = None


def log_line(msg):
    print(msg, flush=True)
    if LOG_FILE is not None:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def normalize(text):
    return " ".join(text.split())


def tokenize_latex(s):
    return TOKEN_RE.findall(s)[:MAX_TOK]


def levenshtein(a, b):
    """Vectorized row-DP edit distance over sequences (str or token list)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if isinstance(a, str):
        a_arr = np.frombuffer(a.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
        b_arr = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    else:
        vocab = {t: i for i, t in enumerate(set(a) | set(b))}
        a_arr = np.array([vocab[t] for t in a], dtype=np.int64)
        b_arr = np.array([vocab[t] for t in b], dtype=np.int64)
    n = len(b_arr)
    idx = np.arange(n + 1, dtype=np.int64)
    prev = idx.copy()
    for i in range(len(a_arr)):
        cand = np.empty(n + 1, dtype=np.int64)
        cand[0] = i + 1
        cand[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b_arr != a_arr[i]))
        running = np.minimum.accumulate(cand - idx)
        prev = np.minimum(cand, running + idx)
    return int(prev[-1])


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ta in a:
        cur = [0] * (len(b) + 1)
        for j, tb in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if ta == tb else (prev[j] if prev[j] >= cur[j - 1] else cur[j - 1])
        prev = cur
    return prev[-1]


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)) if len(tokens) >= n else Counter()


def bleu(ref_tokens, hyp_tokens, max_n=4):
    if not hyp_tokens or not ref_tokens:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hyp_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        overlap = sum(min(c, ref_ng.get(g, 0)) for g, c in hyp_ng.items())
        total = sum(hyp_ng.values())
        # epsilon smoothing so a single missing order doesn't zero out BLEU
        p_n = (overlap + 1e-9) / (total + 1e-9)
        log_sum += np.log(p_n)
    geo = np.exp(log_sum / max_n)
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens)))
    return float(bp * geo)


def rouge_l(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    l = lcs_len(ref_tokens, hyp_tokens)
    if l == 0:
        return 0.0
    p = l / len(hyp_tokens)
    r = l / len(ref_tokens)
    return float(2 * p * r / (p + r))


def bag_f1(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    rc, hc = Counter(ref_tokens), Counter(hyp_tokens)
    tp = sum((rc & hc).values())
    if tp == 0:
        return 0.0
    p = tp / sum(hc.values())
    r = tp / sum(rc.values())
    return float(2 * p * r / (p + r))


def struct_ok(text):
    return (
        "\\documentclass" in text and "\\end{document}" in text
        and text.count("{") == text.count("}")
        and text.count("\\begin") == text.count("\\end")
    )


def chrf(ref, hyp, max_n=6, beta=2.0):
    """chrF: character n-gram F-score (F_beta over char 1..6-grams). Standard MT/OCR
    metric, robust to tokenization. Higher is better."""
    ref, hyp = ref.strip(), hyp.strip()
    if not ref or not hyp:
        return 0.0
    precs, recs = [], []
    for n in range(1, max_n + 1):
        rg = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        hg = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        if not rg or not hg:
            continue
        overlap = sum((rg & hg).values())
        precs.append(overlap / sum(hg.values()))
        recs.append(overlap / sum(rg.values()))
    if not precs:
        return 0.0
    P, R = sum(precs) / len(precs), sum(recs) / len(recs)
    if P + R == 0:
        return 0.0
    b2 = beta * beta
    return float((1 + b2) * P * R / (b2 * P + R))


def compile_tex(tex_str, out_pdf=None, timeout=30):
    """Compile LaTeX with pdflatex. Returns (success, pdf_bytes). If out_pdf is
    given and compilation succeeds, the PDF is also written there."""
    if not tex_str or "\\documentclass" not in tex_str:
        return False, None
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "d.tex").write_text(tex_str, encoding="utf-8")
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "d.tex"],
                cwd=td, capture_output=True, timeout=timeout,
            )
        except Exception:
            return False, None
        pdf = Path(td) / "d.pdf"
        if pdf.exists():
            data = pdf.read_bytes()
            if out_pdf:
                Path(out_pdf).write_bytes(data)
            return True, data
        return False, None


_SPACE_CMDS = re.compile(r"\\(?:,|;|:|!|quad|qquad|enspace|thinspace|;|\s)")


def canon_latex(s):
    """Light canonicalization to remove harmless stylistic differences before
    edit distance, so a model isn't penalized for \\dfrac-vs-\\frac etc. Compares
    the document body only, unifies equivalent commands, and drops cosmetic
    spacing/whitespace. This is the reliable, style-robust counterpart to CER."""
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", s, re.DOTALL)
    if m:
        s = m.group(1)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)                       # dfrac/tfrac -> frac
    s = re.sub(r"\\(?:left|right|bigl|bigr|biggl|biggr|big|Big|bigg|Bigg)\b", "", s)
    s = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle)\b", "", s)
    s = _SPACE_CMDS.sub(" ", s)                                   # cosmetic spacing commands
    s = s.replace(r"\\", " ")                                     # line breaks
    s = re.sub(r"\\(?:begin|end)\{(?:align|equation|gather|displaymath|math)\*?\}", "", s)
    s = re.sub(r"\\[\[\]()]", "", s)                              # \[ \] \( \) delimiters
    s = s.replace("$", "")                                        # inline/display $ delimiters
    s = re.sub(r"\s+", "", s)                                     # ignore all whitespace
    return s


def score(ref, hyp):
    nref, nhyp = normalize(ref), normalize(hyp)
    dist = levenshtein(nref, nhyp)
    cer = dist / max(1, len(nref))
    rt, ht = tokenize_latex(nref), tokenize_latex(nhyp)
    tok_dist = levenshtein(rt, ht)
    tok_err = tok_dist / max(1, len(rt))
    cref, chyp = canon_latex(ref), canon_latex(hyp)
    ncer = levenshtein(cref, chyp) / max(1, len(cref))
    return {
        "cer": round(cer, 4),
        "ncer": round(ncer, 4),
        "charsim": round(max(0.0, 1 - cer), 4),
        "tok_err": round(tok_err, 4),
        "bleu": round(bleu(rt, ht), 4),
        "rouge_l": round(rouge_l(rt, ht), 4),
        "math_f1": round(bag_f1(rt, ht), 4),
        "chrf": round(chrf(nref, nhyp), 4),
        "struct": struct_ok(hyp),
        "exact": nref == nhyp,
        "len_rate": round(len(nhyp) / max(1, len(nref)), 3),
    }


def summarize(results):
    if not results:
        return {}
    def mean(k):
        vals = [r[k] for r in results if k in r and r[k] is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    cers = [r["cer"] for r in results]
    comp = [r["compile_ok"] for r in results if r.get("compile_ok") is not None]
    return {
        "samples": len(results),
        "mean_cer": mean("cer"),
        "median_cer": round(float(np.median(cers)), 4),
        "norm_cer": mean("ncer"),
        "charsim": mean("charsim"),
        "tok_err": mean("tok_err"),
        "bleu": mean("bleu"),
        "rouge_l": mean("rouge_l"),
        "math_f1": mean("math_f1"),
        "chrf": mean("chrf"),
        "struct_pct": round(100.0 * sum(r["struct"] for r in results) / len(results), 1),
        "compile_rate": round(100.0 * sum(comp) / len(comp), 1) if comp else None,
        "cer_under_10pct": round(100.0 * sum(c < 0.10 for c in cers) / len(cers), 1),
        "cer_under_30pct": round(100.0 * sum(c < 0.30 for c in cers) / len(cers), 1),
        "len_rate": mean("len_rate"),
        "avg_latency_s": round(float(np.mean([r["latency_s"] for r in results])), 2),
    }


def cap_image_tokens(processor):
    ip = processor.image_processor
    patch = getattr(ip, "patch_size", 14)
    merge = getattr(ip, "merge_size", 2)
    max_pixels = MAX_IMAGE_TOKENS * (patch * patch) * (merge * merge)
    try:
        ip.size["longest_edge"] = int(max_pixels)
    except Exception:
        pass


def generate(model, processor, image, max_new_tokens, device, rep_penalty=1.0, no_repeat_ngram=0):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if rep_penalty and rep_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = rep_penalty
    if no_repeat_ngram and no_repeat_ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    dt = time.time() - t0
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip(), dt


def generate_batch(model, processor, images, max_new_tokens, device, rep_penalty=1.0, no_repeat_ngram=0):
    """Greedy-generate LaTeX for several images at once (left-padded). Validated to
    produce byte-identical output to single-page generation at batch<=4."""
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}],
        tokenize=False, add_generation_prompt=True,
    )
    old_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt", padding=True).to(device)
        gk = {"max_new_tokens": max_new_tokens, "do_sample": False}
        if rep_penalty and rep_penalty != 1.0:
            gk["repetition_penalty"] = rep_penalty
        if no_repeat_ngram and no_repeat_ngram > 0:
            gk["no_repeat_ngram_size"] = no_repeat_ngram
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, **gk)
        dt = time.time() - t0
        g = out[:, inputs["input_ids"].shape[1]:]
        texts = [processor.tokenizer.decode(x, skip_special_tokens=True).strip() for x in g]
    finally:
        processor.tokenizer.padding_side = old_side
    return texts, dt / max(1, len(images))


def generate_chunk(model, processor, images, max_new_tokens, device, rep_penalty, no_repeat_ngram):
    """Batched generation that NEVER raises. On OOM (dense pages) or any other
    error it retries one page at a time; a page that still fails returns "" so
    the benchmark always continues. Returns (texts, avg_seconds_per_page)."""
    def _empty():
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    try:
        return generate_batch(model, processor, images, max_new_tokens, device, rep_penalty, no_repeat_ngram)
    except torch.cuda.OutOfMemoryError:
        _empty()
        log_line(f"  OOM on batch of {len(images)} - retrying one page at a time")
    except Exception as e:
        log_line(f"  batch error ({type(e).__name__}: {str(e)[:80]}) - retrying one page at a time")
        _empty()
    texts, total, n = [], 0.0, 0
    for im in images:
        try:
            t, dt = generate_batch(model, processor, [im], max_new_tokens, device, rep_penalty, no_repeat_ngram)
            texts.append(t[0]); total += dt; n += 1
        except torch.cuda.OutOfMemoryError:
            _empty()
            log_line("  OOM on a single dense page - recording empty output and continuing")
            texts.append("")
        except Exception as e:
            log_line(f"  single-page error ({type(e).__name__}) - recording empty output")
            texts.append("")
    return texts, (total / n if n else 0.0)


def model_label(spec):
    if spec == "base":
        return "Base GLM-OCR"
    p = Path(spec)
    return p.parent.name if p.name == "final" else p.name


def load_model(spec, device):
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype=torch.bfloat16)
    if spec != "base":
        model = PeftModel.from_pretrained(model, spec)
    model.to(device).eval()
    return model


def safe_label(spec):
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_label(spec))


def cache_file(cache_dir, spec):
    """Per-model result cache, keyed by a hash of the spec so re-running the same
    model resumes regardless of position in the comparison list."""
    h = hashlib.md5(spec.encode("utf-8")).hexdigest()[:8]
    return cache_dir / f"{safe_label(spec)}__{h}.jsonl"


def load_cache(cache_path):
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["id"]] = r
            except Exception:
                pass  # skip a half-written final line from a power cut
    return done


def append_result(cache_path, row):
    """Append one scored sample and fsync it to disk before returning, so a power
    cut loses at most the single page currently being generated."""
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _row_done(row, is_base):
    """A cached page is resumable only if it has the saved generation and, for
    non-base models, the compile/CDM fields too."""
    if not row or "hyp" not in row:
        return False
    if not is_base and "compile_ok" not in row:
        return False
    return True


def evaluate(model, processor, samples, max_new_tokens, device, key, label, out_path,
             partial, cache_path, out_dir, is_base, rep_penalty, no_repeat_ngram, batch_size):
    from concurrent.futures import ThreadPoolExecutor
    done = load_cache(cache_path)
    results = [None] * len(samples)
    todo = []  # (index, sample) for pages still needing work
    for idx, s in enumerate(samples):
        if _row_done(done.get(s["id"]), is_base):
            results[idx] = done[s["id"]]
        else:
            todo.append((idx, s))
    processed = len(samples) - len(todo)
    if processed:
        log_line(f"{label}: resuming - {processed}/{len(samples)} already cached")
    # pdflatex runs as subprocesses (release the GIL), so compile the whole batch
    # in parallel threads instead of serially blocking the next batch
    compile_pool = None if is_base else ThreadPoolExecutor(max_workers=max(2, batch_size))
    try:
        for bstart in range(0, len(todo), batch_size):
            chunk = todo[bstart:bstart + batch_size]
            try:
                # 1) gather text: reuse cached hyp where present, else generate
                hyps, per = {}, 0.0
                for idx, s in chunk:
                    c = done.get(s["id"])
                    if c and "hyp" in c:
                        hyps[idx] = c["hyp"]
                need = [(idx, s) for idx, s in chunk if idx not in hyps]
                if need:
                    imgs = []
                    for _, s in need:
                        try:
                            imgs.append(Image.open(DATASET_DIR / s["image"]).convert("RGB"))
                        except Exception as e:
                            log_line(f"{label} image load failed {s['id']}: {e} - using blank")
                            imgs.append(Image.new("RGB", (224, 224), (255, 255, 255)))
                    texts, per = generate_chunk(model, processor, imgs, max_new_tokens, device, rep_penalty, no_repeat_ngram)
                    for (idx, _), t in zip(need, texts):
                        hyps[idx] = t
                need_idx = {i for i, _ in need}
                # 2) score + write .tex; submit compiles in parallel
                rows, futs = {}, {}
                for idx, s in chunk:
                    ref = next(c["content"] for c in s["conversations"] if c["role"] == "assistant")
                    hyp = hyps.get(idx, "")
                    row = score(ref, hyp)
                    row["id"] = s["id"]
                    row["latency_s"] = round(per, 2) if idx in need_idx else done.get(s["id"], {}).get("latency_s", 0.0)
                    row["hyp"] = hyp
                    try:
                        (out_dir / f"{s['id']}.tex").write_text(hyp or "", encoding="utf-8")
                    except Exception as e:
                        log_line(f"{label} tex write failed {s['id']}: {e}")
                    if is_base:
                        row["compile_ok"] = None
                    else:
                        futs[idx] = compile_pool.submit(compile_tex, hyp, out_dir / f"{s['id']}.pdf")
                    rows[idx] = row
                # 3) collect compile results (compile_tex is itself exception-safe)
                for idx, fut in futs.items():
                    try:
                        ok, _ = fut.result()
                    except Exception:
                        ok = False
                    rows[idx]["compile_ok"] = ok
                # 4) persist + update progress
                for idx, s in chunk:
                    append_result(cache_path, rows[idx])
                    results[idx] = rows[idx]
                    processed += 1
                partial["models"][key]["summary"] = summarize([r for r in results if r])
                partial["progress"] = {"model": label, "key": key, "done": processed, "total": len(samples)}
                out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
                first = rows[chunk[0][0]]
                cstr = "" if is_base else f" compile={first['compile_ok']}"
                log_line(f"{label} [{processed}/{len(samples)}] batch@{chunk[0][1]['id']} cer={first['cer']:.3f} ncer={first['ncer']:.3f}{cstr} ~{per:.1f}s/pg")
            except Exception as e:
                # a single batch failing must never kill the model's whole run
                processed += len(chunk)
                log_line(f"{label} BATCH FAILED @{chunk[0][1]['id']} ({type(e).__name__}: {str(e)[:100]}) - skipping {len(chunk)} pages")
    finally:
        if compile_pool:
            compile_pool.shutdown(wait=True)
    return [r for r in results if r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help='2-3 specs: "base" or adapter dir paths')
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--fresh", action="store_true", help="ignore any cached results and start over")
    ap.add_argument("--rep-penalty", type=float, default=1.0, help="repetition_penalty (>1 curbs over-generation/loops)")
    ap.add_argument("--no-repeat-ngram", type=int, default=0, help="block repeating n-grams (e.g. 3) to stop loops")
    ap.add_argument("--batch-size", type=int, default=4, help="pages generated together per GPU pass (4 is exact + ~2.5-3x faster; >4 can corrupt greedy output)")
    args = ap.parse_args()

    out_path = Path(args.output)
    cache_dir = out_path.parent / "bench_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = out_path.parent / "bench_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    global LOG_FILE
    LOG_FILE = out_path.parent / "benchmark_run.log"
    LOG_FILE.write_text(f"=== benchmark started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n", encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = []
    with open(DATASET_DIR / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[: args.samples]

    keys = [chr(ord("a") + i) for i in range(len(args.models))]
    labels = [model_label(m) for m in args.models]
    log_line(f"Benchmark: {' vs '.join(labels)} on {len(samples)} test samples, device={device}")

    partial = {
        "state": "running",
        "started_at": time.time(),
        "samples": len(samples),
        "model_keys": keys,
        "models": {k: {"label": lbl, "spec": spec, "summary": {}}
                   for k, lbl, spec in zip(keys, labels, args.models)},
    }
    out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    caches = {k: cache_file(cache_dir, spec) for k, spec in zip(keys, args.models)}
    if args.fresh:
        for cp in caches.values():
            if cp.exists():
                cp.unlink()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cap_image_tokens(processor)

    sample_ids = [s["id"] for s in samples]
    all_results = {k: [] for k in keys}  # every key present even if a model fails
    for key, spec, label in zip(keys, args.models, labels):
        model = None
        try:
            is_base = (spec == "base")
            cache_path = caches[key]
            out_dir = outputs_dir / safe_label(spec)
            out_dir.mkdir(parents=True, exist_ok=True)
            done = load_cache(cache_path)
            needed = [sid for sid in sample_ids if not _row_done(done.get(sid), is_base)]
            if not needed:
                # every requested page fully done on disk — no model load (and no GPU) needed
                log_line(f"[{label}] fully cached ({len(sample_ids)} samples) - skipping model load")
                all_results[key] = [done[sid] for sid in sample_ids]
                partial["models"][key]["summary"] = summarize(all_results[key])
                out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
                continue
            # only load the model if pages still need GENERATION (vs re-deriving metrics)
            need_gen = any("hyp" not in (done.get(sid) or {}) for sid in needed)
            if need_gen:
                log_line(f"Loading [{label}] ({spec}) - {len(needed)} of {len(sample_ids)} pages to process...")
                model = load_model(spec, device)
            else:
                log_line(f"[{label}] text already saved - recomputing metrics for {len(needed)} pages (no GPU)")
            all_results[key] = evaluate(
                model, processor, samples, args.max_new_tokens, device, key, label, out_path,
                partial, cache_path, out_dir, is_base, args.rep_penalty, args.no_repeat_ngram, args.batch_size,
            )
        except Exception as e:
            # one model failing (bad adapter, load error, etc.) must not abort the others
            import traceback
            log_line(f"[{label}] MODEL FAILED ({type(e).__name__}: {str(e)[:120]}) - continuing to next model")
            log_line(traceback.format_exc()[-600:])
            # keep whatever pages were already cached for this model so the run isn't a total loss
            try:
                all_results[key] = [done[sid] for sid in sample_ids if _row_done(done.get(sid), spec == "base")]
            except Exception:
                all_results[key] = []
        finally:
            if model is not None:
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    summaries = {k: summarize(all_results[k]) for k in keys}
    base_cer = summaries[keys[0]].get("mean_cer")
    best_key = min(keys, key=lambda k: summaries[k].get("mean_cer") if summaries[k].get("mean_cer") is not None else 1e9)
    improvement = None
    if base_cer and summaries[best_key].get("mean_cer") is not None:
        improvement = {
            "best_key": best_key,
            "best_label": labels[keys.index(best_key)],
            "ref_label": labels[0],
            "cer_reduction_pct": round(100.0 * (base_cer - summaries[best_key]["mean_cer"]) / max(1e-9, base_cer), 1),
        }

    final = {
        "state": "done",
        "finished_at": time.time(),
        "samples": len(samples),
        "model_keys": keys,
        "models": {k: {"label": labels[i], "spec": args.models[i], "summary": summaries[k]}
                   for i, k in enumerate(keys)},
        "improvement": improvement,
        "per_sample": all_results,
    }
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    log_line("BENCHMARK DONE")
    log_line(json.dumps({labels[i]: summaries[k] for i, k in enumerate(keys)}, indent=2))


if __name__ == "__main__":
    main()
