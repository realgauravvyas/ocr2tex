"""Run Baidu Unlimited-OCR on the GLM-OCR test set and save per-page outputs,
reformatted into the SAME shape GLM-OCR/v4.1 produces (a \\documentclass ...
\\end{document} LaTeX document of the handwritten content only), so the two are
comparable on CER / Norm-CER / BLEU / Compile% etc.

Unlimited-OCR is a whole-page document-OCR model: it transcribes printed
headers, footers and page numbers that v4.1 was trained to omit. Its raw output
tags every region by type (<|det|>header/text/equation/footer/page_number ...),
so we keep only the content regions and drop the printed furniture, then wrap
the result in the GLM-OCR LaTeX envelope. Runs in the BaiduOCR venv.
"""
import os, sys, json, time, argparse, tempfile, shutil, re, io, contextlib
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from pathlib import Path

MODEL_NAME = "baidu/Unlimited-OCR"
TEST = Path(r"D:\ocr2tex\workspace\9_split\test.jsonl")
IMG = Path(r"D:\ocr2tex\workspace\9_split\images")
OUT = Path(r"D:\ocr2tex\output\bench_outputs\Baidu_Unlimited-OCR")
PROGRESS = Path(r"D:\ocr2tex\output\baidu_gen_progress.json")  # live status for dashboard


def write_progress(**kw):
    try:
        cur = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
        cur.update(kw)
        cur["updated_at"] = time.time()
        PROGRESS.write_text(json.dumps(cur), encoding="utf-8")
    except Exception:
        pass

PREAMBLE = ("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n"
            "\\usepackage{amsfonts}\n\n\\begin{document}\n\n")
POSTAMBLE = "\n\n\\end{document}\n"

# Printed page furniture v4.1 was trained to ignore.
DROP_TYPES = {"header", "footer", "page_number", "page-number", "pagenumber"}
DET_RE = re.compile(r"<\|det\|>\s*([a-z_\-]+)\s*\[[0-9,\s]*\]\s*<\|/det\|>", re.I)
# Printed strings that leak when mis-typed as `text` (every exam page has them).
FURNITURE_RE = re.compile(r"(Space\s+for\s+answering\s+Question|^\s*Q\s*-?\s*\d+\s*\)|^\s*\d+\s+of\s+\d+\s*$)", re.I)


def _drop_furniture_lines(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not FURNITURE_RE.search(ln.strip()))


def strip_tokens(t: str) -> str:
    t = re.sub(r"<\|det\|>.*?<\|/det\|>", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|/?[a-z_]+\|>", "", t)
    t = re.sub(r"!\[\]\(images/\d+\.jpg\)", "", t)   # image placeholders
    return t


def to_glm_format(raw: str) -> str:
    """Turn Unlimited-OCR output into a GLM-OCR-style LaTeX document."""
    raw = (raw or "").strip()
    parts = DET_RE.split(raw)   # [pre, type1, content1, type2, content2, ...]
    if len(parts) >= 3:
        body_chunks = []
        for i in range(1, len(parts), 2):
            rtype = parts[i].lower()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            if rtype in DROP_TYPES:
                continue
            content = _drop_furniture_lines(strip_tokens(content)).strip()
            if content:
                body_chunks.append(content)
        body = "\n\n".join(body_chunks)
    else:
        # No region tags — fall back to line heuristics on cleaned text.
        body = _drop_furniture_lines(strip_tokens(raw))
    body = re.sub(r"^#{1,6}\s*", "", body, flags=re.M)     # markdown headers
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return PREAMBLE + body + POSTAMBLE


def load_model():
    import torch
    try:
        from transformers.utils.import_utils import is_torch_fx_available  # noqa
    except ImportError:
        import transformers.utils.import_utils as _iutils
        import transformers.utils as _tu
        _iutils.is_torch_fx_available = lambda: False
        _tu.is_torch_fx_available = lambda: False
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      use_safetensors=True, torch_dtype=torch.bfloat16)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tok


def run_one(model, tok, img: Path, prompt: str):
    """Capture the streamer's det-tagged generation from stdout (result.md has the
    region tags stripped, which we need to drop printed furniture)."""
    tmp = Path(tempfile.mkdtemp())
    buf = io.StringIO()
    try:
        t1 = time.time()
        with contextlib.redirect_stdout(buf):
            model.infer(tok, prompt=prompt, image_file=str(img), output_path=str(tmp),
                        base_size=1024, image_size=640, crop_mode=True, max_length=8192,
                        no_repeat_ngram_size=35, ngram_window=128, save_results=True)
        dt = time.time() - t1
        streamed = buf.getvalue().split("save results:")[0]
        if "<|det|>" not in streamed:          # fallback: cleaned markdown
            md = tmp / "result.md"
            streamed = md.read_text(encoding="utf-8", errors="replace") if md.exists() else ""
        return streamed, dt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=700)
    ap.add_argument("--prompt", default="<image>document parsing.")
    ap.add_argument("--page-timeout", type=float, default=120.0, help="hard wall-clock cap (s) on generation per page")
    ap.add_argument("--probe", action="store_true", help="print raw+formatted for inspection, no save")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()][:args.samples]

    def _done(r):  # done = non-trivial .tex exists (re-runs stale 0-byte / partial writes)
        f = OUT / f"{r['id']}.tex"
        return f.exists() and f.stat().st_size >= 50

    pending = rows if args.probe else [r for r in rows if not _done(r)]
    total = len(rows)
    existing = total - len(pending)
    print(f"Unlimited-OCR: {total} target, {len(pending)} to run | prompt={args.prompt!r}", flush=True)
    if not args.probe:
        write_progress(status="loading", model="baidu/Unlimited-OCR", total=total, done=existing,
                       current="", eta_h=None, avg_s=None, recent=[], started_at=time.time())

    print("loading model...", flush=True)
    t0 = time.time()
    model, tok = load_model()
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    # Hard wall-clock cap per page: infer() calls self.generate() internally, so
    # patch generate to inject max_time -> transformers adds a MaxTimeCriteria that
    # stops generation gracefully, returning whatever was produced so far.
    pt = float(args.page_timeout)
    try:
        model.generation_config.max_time = pt
    except Exception:
        pass
    _orig_gen = model.generate
    def _timed_generate(*a, **k):
        k.setdefault("max_time", pt)
        return _orig_gen(*a, **k)
    model.generate = _timed_generate
    print(f"per-page generation timeout: {pt:.0f}s", flush=True)

    if not args.probe:
        write_progress(status="running", load_s=round(time.time() - t0, 1))

    times, recent = [], []
    for i, r in enumerate(pending):
        pid = r["id"]
        try:
            raw, dt = run_one(model, tok, IMG / f"{pid}.png", args.prompt)
            tex = to_glm_format(raw)
        except Exception as e:
            print(f"  {pid} FAIL: {e}", flush=True)
            raw, tex, dt = "", to_glm_format(""), 0.0
        times.append(dt)
        if args.probe:
            print(f"\n===== {pid} ({dt:.1f}s) RAW result.md =====\n{raw[:1200]}")
            print(f"\n----- {pid} FORMATTED (GLM-style) -----\n{tex[:1200]}\n")
        else:
            (OUT / f"{pid}.tex").write_text(tex, encoding="utf-8")
            (OUT / f"{pid}.raw").write_text(raw, encoding="utf-8")
            (OUT / f"{pid}.time").write_text(str(round(dt, 2)), encoding="utf-8")
            avg = sum(times) / len(times)
            eta = avg * (len(pending) - i - 1) / 3600
            done = existing + i + 1
            recent = ([{"pid": pid, "chars": len(tex), "secs": round(dt, 1)}] + recent)[:10]
            print(f"  [{i+1}/{len(pending)}] {pid}: {len(tex)} chars, {dt:.1f}s, ETA {eta:.1f}h", flush=True)
            write_progress(status="running", done=done, total=total, current=pid,
                           pct=round(100 * done / max(1, total), 1), eta_h=round(eta, 2),
                           avg_s=round(avg, 1), recent=recent)

    if not args.probe:
        write_progress(status="done", current="", eta_h=0.0,
                       done=existing + len(pending), total=total)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
