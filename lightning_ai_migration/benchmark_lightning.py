# -*- coding: utf-8 -*-
"""
GLM-OCR v5.0 Dual-Modality Benchmark & LaTeX Verification Suite for Lightning AI Studio.
Evaluates the fine-tuned LoRA adapter on all 700 held-out test samples (data/test.jsonl).

Produces full metric battery:
  - CER (Character Error Rate)
  - Norm-CER (Style-Invariant Canonicalized LaTeX Error)
  - CharSim (1 - CER)
  - TokEdit (Token-level Levenshtein error)
  - BLEU-4 (LaTeX n-gram precision)
  - ROUGE-L (LCS token recall/F1)
  - Math-F1 (Mathematical symbol presence F1)
  - chrF (Character 6-gram F-score)
  - Struct % (% valid matching environments and delimiters)
  - Compile % (% clean pdflatex compilations to valid PDF, if pdflatex available)
  - Latency (Generation seconds per page on A100)
"""

import os
import re
import sys
import json
import time
import argparse
import warnings
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_NAME = "zai-org/GLM-OCR"
USER_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into a complete, compilable LaTeX document. Ignore printed text, "
    "student info, page numbers, cancelled work and rough work. Output only LaTeX."
)

TOKEN_RE = re.compile(r"\\[a-zA-Z]+\*?|\\.|[{}()\[\]^_&+=\-*/<>|]|[a-zA-Z0-9]+|\S")
SPACE_CMDS = re.compile(r"\\(?:,|;|:|!|quad|qquad|enspace|thinspace|;|\s)")
MAX_TOK = 1200


def log(msg):
    print(msg, flush=True)


def normalize(text):
    return " ".join(text.split())


def tokenize_latex(s):
    return TOKEN_RE.findall(s)[:MAX_TOK]


def levenshtein(a, b):
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


def rouge_l(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = lcs_len(ref_tokens, hyp_tokens)
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


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
        p_n = (overlap + 1e-4) / (max(1, len(hyp_tokens) - n + 1) + 1e-4)
        log_sum += (1.0 / max_n) * np.log(p_n)
    bp = min(1.0, np.exp(1.0 - len(ref_tokens) / max(1, len(hyp_tokens))))
    return float(bp * np.exp(log_sum))


def bag_f1(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    rc = Counter(ref_tokens)
    hc = Counter(hyp_tokens)
    common = sum((rc & hc).values())
    p = common / len(hyp_tokens)
    r = common / len(ref_tokens)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def chrf(ref, hyp, max_n=6, beta=2.0):
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
    return (1 + b2) * P * R / (b2 * P + R)


def struct_ok(text):
    return (
        "\\documentclass" in text and "\\end{document}" in text
        and text.count("{") == text.count("}")
        and text.count("\\begin") == text.count("\\end")
    )


def canon_latex(s):
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", s, re.DOTALL)
    if m:
        s = m.group(1)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\(?:left|right|bigl|bigr|biggl|biggr|big|Big|bigg|Bigg)\b", "", s)
    s = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle)\b", "", s)
    s = SPACE_CMDS.sub(" ", s)
    s = s.replace(r"\\", " ")
    s = re.sub(r"\\(?:begin|end)\{(?:align|equation|gather|displaymath|math)\*?\}", "", s)
    s = re.sub(r"\\[\[\]()]", "", s)
    s = s.replace("$", "")
    s = re.sub(r"\s+", "", s)
    return s


def compile_tex(tex_str, out_pdf=None, timeout=25):
    if not tex_str or "\\documentclass" not in tex_str:
        return False, None
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "doc.tex").write_text(tex_str, encoding="utf-8")
        try:
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "doc.tex"],
                cwd=td, capture_output=True, timeout=timeout,
            )
            if res.returncode != 0:
                return False, None
        except Exception:
            return False, None
        pdf = Path(td) / "doc.pdf"
        if pdf.exists():
            data = pdf.read_bytes()
            if out_pdf:
                try:
                    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_pdf).write_bytes(data)
                except Exception:
                    pass
            return True, data
        return False, None


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


def find_latest_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    final_dir = ckpt_dir / "final"
    if (final_dir / "adapter_model.safetensors").exists():
        return final_dir
    ckpts = sorted(
        ckpt_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        reverse=True
    )
    for c in ckpts:
        if (c / "adapter_model.safetensors").exists() and (c / "adapter_model.safetensors").stat().st_size > 1024 * 512:
            return c
    return None


def generate_single(model, processor, image, max_new_tokens, device):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.time() - t0
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip(), dt


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR v5.0 Benchmark Suite on Full Held-Out Test Set")
    parser.add_argument("--dataset-dir", default="./data", help="Path to split dataset directory containing test.jsonl")
    parser.add_argument("--image-base", default="./data", help="Path to dataset images base")
    parser.add_argument("--output-dir", default="./output", help="Training output directory containing checkpoints")
    parser.add_argument("--adapter", default=None, help="Specific adapter checkpoint path (defaults to latest or ./output/final)")
    parser.add_argument("--limit", type=int, default=None, help="Number of test samples to benchmark (default: None = all 700 test pages)")
    parser.add_argument("--output-json", default="./output/benchmark_results_v5.json", help="Summary JSON output path")
    parser.add_argument("--out-dir", default="./output/bench_outputs", help="Directory to save generated .tex files")
    args = parser.parse_args()

    test_file = Path(args.dataset_dir) / "test.jsonl"
    image_base = Path(args.image_base)
    output_dir = Path(args.output_dir)

    if not test_file.exists():
        log(f"[ERROR] Test split not found at: {test_file}")
        sys.exit(1)

    adapter_path = Path(args.adapter) if args.adapter else find_latest_checkpoint(output_dir)
    if not adapter_path:
        log(f"[ERROR] No trained checkpoint/final adapter found in: {output_dir}")
        sys.exit(1)

    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    total_test_samples = len(samples)
    if args.limit and args.limit < total_test_samples:
        samples = samples[:args.limit]

    log(f"\n========================================================")
    log(f"GLM-OCR v5.0 Official Benchmark Suite (A100 Accelerated)")
    log(f"Test Split Dataset   : {test_file} ({total_test_samples} total test pages)")
    log(f"Benchmarking Samples : {len(samples)} pages")
    log(f"Evaluating Adapter   : {adapter_path}")
    log(f"========================================================\n")

    has_pdflatex = subprocess.run(["which", "pdflatex"], capture_output=True).returncode == 0
    if not has_pdflatex:
        log("[INFO] pdflatex not installed in environment. Compile % will be skipped (Struct %, CER, BLEU, Math-F1 fully evaluated).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading Base GLM-OCR Processor and Model ({device})...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    log(f"Attaching Fine-Tuned LoRA Adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path)).to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "cache_eval.jsonl"

    cached = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cached[item["id"]] = item

    results = []
    pool = ThreadPoolExecutor(max_workers=4)
    compile_futures = {}

    log(f"Starting Generation across all {len(samples)} test pages on A100...\n")
    t_start = time.time()

    for i, s in enumerate(samples, 1):
        sid = s["id"]
        ref = next(c["content"] for c in s["conversations"] if c["role"] == "assistant")
        
        if sid in cached:
            row = cached[sid]
            results.append(row)
            log(f"  [{i:3d}/{len(samples)}] {sid} (Cached) | CER={row['cer']:.3f} | NCER={row['ncer']:.3f} | Struct={'OK' if row['struct'] else 'ERR'}")
            continue

        img_rel = s["image"]
        img_path = image_base / img_rel
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log(f"  [WARN] Failed reading {img_path}: {e}")
            img = Image.new("RGB", (224, 224), (255, 255, 255))

        hyp, latency = generate_single(model, processor, img, max_new_tokens=1024, device=device)
        row = score(ref, hyp)
        row["id"] = sid
        row["latency_s"] = round(latency, 2)
        row["hyp"] = hyp

        tex_file = out_dir / f"{sid}.tex"
        tex_file.write_text(hyp, encoding="utf-8")

        if has_pdflatex:
            pdf_file = out_dir / f"{sid}.pdf"
            compile_futures[sid] = pool.submit(compile_tex, hyp, pdf_file)
        else:
            row["compile_ok"] = None

        results.append(row)
        with open(cache_file, "a", encoding="utf-8") as cf:
            cf.write(json.dumps(row) + "\n")

        log(f"  [{i:3d}/{len(samples)}] {sid} | CER={row['cer']:.3f} | NCER={row['ncer']:.3f} | Struct={'OK' if row['struct'] else 'ERR'} | {latency:.2f}s")

    if has_pdflatex:
        log("\nCollecting pdflatex compilation results...")
        for sid, fut in compile_futures.items():
            ok, _ = fut.result()
            for r in results:
                if r["id"] == sid:
                    r["compile_ok"] = ok
                    break

    summary = summarize(results)
    tot_time = time.time() - t_start

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 75)
    log(f"       GLM-OCR v5.0 OFFICIAL BENCHMARK SCORECARD ({len(samples)} PAGES)       ")
    log("=" * 75)
    log(f"  Test Pages Evaluated   : {summary.get('samples')} (Full held-out test split)")
    log(f"  Mean CER               : {summary.get('mean_cer', 0):.4f} (Median: {summary.get('median_cer', 0):.4f})")
    log(f"  Normalized CER (NCER)  : {summary.get('norm_cer', 0):.4f}")
    log(f"  Character Similarity   : {summary.get('charsim', 0)*100:.1f}%")
    log(f"  Structure Validity     : {summary.get('struct_pct', 0):.1f}% (Brackets & \\begin/\\end)")
    if summary.get('compile_rate') is not None:
        log(f"  PDF Compile Success    : {summary.get('compile_rate', 0):.1f}%")
    log(f"  BLEU-4 Precision       : {summary.get('bleu', 0):.4f}")
    log(f"  ROUGE-L Token F1       : {summary.get('rouge_l', 0):.4f}")
    log(f"  Math-F1 Symbol Score   : {summary.get('math_f1', 0):.4f}")
    log(f"  chrF n-gram F-score    : {summary.get('chrf', 0):.4f}")
    log(f"  CER < 10% Accuracy     : {summary.get('cer_under_10pct', 0):.1f}% of pages")
    log(f"  CER < 30% Usability    : {summary.get('cer_under_30pct', 0):.1f}% of pages")
    log(f"  Average Page Latency   : {summary.get('avg_latency_s', 0):.2f}s on A100")
    log(f"  Total Benchmark Time   : {tot_time/60:.1f} minutes")
    log("=" * 75)
    log(f"\n[SUCCESS] Full benchmark metrics saved to: {args.output_json}")
    log(f"All 700 generated LaTeX files saved to: {args.out_dir}/\n")

if __name__ == "__main__":
    main()
