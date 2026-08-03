"""Run GLM-OCR (base + adapters) on a single image for the dashboard Inspector.

Loads the base model once, then applies each LoRA adapter in turn, generating
LaTeX for the given image. Saves <model_label>.tex for every model and a
compiled <model_label>.pdf for fine-tuned models (base is text-only). Writes
result.json for the dashboard.

Called as a subprocess by app.py when a user uploads a new image.
"""

import re
import json
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_NAME = "zai-org/GLM-OCR"
MAX_IMAGE_TOKENS = 1536
USER_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into a complete, compilable LaTeX document. Ignore printed text, "
    "student info, page numbers, cancelled work and rough work. Output only LaTeX."
)


def log(m):
    print(m, flush=True)


def model_label(spec):
    if spec == "base":
        return "Base GLM-OCR"
    p = Path(spec)
    return p.parent.name if p.name == "final" else p.name


def safe_label(spec):
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_label(spec))


def cap_image_tokens(processor):
    ip = processor.image_processor
    patch = getattr(ip, "patch_size", 14)
    merge = getattr(ip, "merge_size", 2)
    try:
        ip.size["longest_edge"] = int(MAX_IMAGE_TOKENS * (patch * patch) * (merge * merge))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--no-prefilter", action="store_true", help="disable the printed-only pre-filter (show raw model output)")
    ap.add_argument("--bench-out", type=str, default="", help="if set, overwrite <bench-out>/<slug>/<page-id>.{tex,pdf} in place")
    ap.add_argument("--page-id", type=str, default="", help="page id for bench-layout overwrite")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_mode = bool(args.bench_out and args.page_id)

    def targets(slug):
        # where this model's output files go: bench layout (overwrite in place) or the temp out dir
        if bench_mode:
            d = Path(args.bench_out) / slug
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{args.page_id}.tex", d / f"{args.page_id}.pdf"
        return out_dir / f"{slug}.tex", out_dir / f"{slug}.pdf"

    def write_out(slug, hyp, pdf_bytes=None):
        # write/overwrite tex; write pdf if given, else remove any stale pdf
        tex_p, pdf_p = targets(slug)
        tex_p.write_text(hyp or "", encoding="utf-8")
        if pdf_bytes:
            pdf_p.write_bytes(pdf_bytes)
        elif pdf_p.exists():
            pdf_p.unlink()  # remove previously-saved pdf (printed-only / base / failed compile)

    # import the benchmark's pdflatex compiler so behavior matches exactly
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_glm_ocr import compile_tex

    result = {"state": "running", "started_at": time.time(),
              "rep_penalty": args.rep_penalty, "no_repeat_ngram": args.no_repeat_ngram,
              "has_handwriting": None, "prefiltered": False, "models": {}}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    image = Image.open(args.image).convert("RGB")
    log(f"Loaded image {args.image}, device={device}")

    # ---- pre-filter: detect printed-only / blank pages and suppress output ----
    # (printed pages should yield EMPTY per the project spec; the models otherwise
    #  transcribe printed text because the silver-standard labels did)
    has_hw = True
    if not args.no_prefilter:
        try:
            from classify_handwriting import classify, KEYS, API_BASE
            from openai import OpenAI
            v = classify(OpenAI(base_url=API_BASE, api_key=KEYS[0]), Path(args.image))
            has_hw = (v != "NO")
            result["has_handwriting"] = has_hw
            log(f"Pre-filter: handwriting={'YES' if has_hw else 'NO'} (verdict={v})")
        except Exception as e:
            log(f"Pre-filter classification failed ({type(e).__name__}) - failing open, will generate")
            result["has_handwriting"] = None

    if result.get("has_handwriting") is False:
        # printed-only page: the correct output is empty for every model. Skip the
        # GPU entirely and write empty .tex; no PDF.
        result["prefiltered"] = True
        for spec in args.models:
            slug = safe_label(spec)
            write_out(slug, "")  # empty tex, removes any stale pdf
            result["models"][slug] = {"label": model_label(spec), "latency_s": 0.0, "chars": 0,
                                      "pdf": False, "suppressed": True}
        result["state"] = "done"
        result["finished_at"] = time.time()
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log("PREFILTER: printed-only page - output suppressed (empty) for all models. INFER DONE")
        return

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cap_image_tokens(processor)

    log("Loading base model...")
    base = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype=torch.bfloat16)
    base.to(device).eval()

    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    if args.rep_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = args.rep_penalty
    if args.no_repeat_ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram

    for spec in args.models:
        label = model_label(spec)
        slug = safe_label(spec)
        is_base = (spec == "base")
        log(f"Running [{label}]...")
        model = base if is_base else PeftModel.from_pretrained(base, spec)
        model.eval()
        try:
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            dt = time.time() - t0
            gen = out[0][inputs["input_ids"].shape[1]:]
            hyp = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
        except Exception as e:
            log(f"[{label}] generation failed: {e}")
            hyp, dt = "", 0.0
        entry = {"label": label, "latency_s": round(dt, 2), "chars": len(hyp), "pdf": False}
        if is_base:
            write_out(slug, hyp)  # tex only (base has no pdf)
        else:
            ok, pdf = compile_tex(hyp)
            write_out(slug, hyp, pdf if ok else None)  # write pdf if it compiled, else drop stale pdf
            entry["pdf"] = ok
            entry["compile_ok"] = ok
        result["models"][slug] = entry
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"[{label}] done: {len(hyp)} chars, {dt:.1f}s" + ("" if is_base else f", compiled={entry['pdf']}"))
        if not is_base:
            model = model.unload() if hasattr(model, "unload") else None
            if device == "cuda":
                torch.cuda.empty_cache()

    result["state"] = "done"
    result["finished_at"] = time.time()
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log("INFER DONE")


if __name__ == "__main__":
    main()
