"""Run the FINE-TUNED Baidu model (base Unlimited-OCR + the LoRA adapter from
finetune/) on the GLM-OCR test set and save per-page outputs for benchmarking,
exactly like the base-Baidu pipeline so it becomes another comparable column.

The fine-tune was trained to emit a complete GLM-style LaTeX document directly
(prompt = the training prompt, target = the reference LaTeX), so we merge the
adapter into the base and call infer() with that same prompt; result.md is the
LaTeX (no det tags). Runs in the BaiduOCR venv. GPU — run only when free.
"""
import os, sys, json, time, argparse, tempfile, shutil
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

sys.path.insert(0, r"D:\ocr2tex\dashboard")
from baidu_format import ft_format

MODEL_NAME = "baidu/Unlimited-OCR"
FT_PROMPT = "<image>Convert the handwriting to a complete LaTeX document."  # must match training
TEST = Path(r"D:\ocr2tex\workspace\9_split\test.jsonl")
IMG = Path(r"D:\ocr2tex\workspace\9_split\images")
OUT = Path(r"D:\ocr2tex\output\bench_outputs\Baidu_OCR_FT_v2_best")
PROGRESS = Path(r"D:\ocr2tex\output\baidu_ft_v2_best_gen_progress.json")
DEFAULT_ADAPTER = Path(r"D:\Claude Code\BaiduOCR\finetune\baidu-ocr-math-v1")


def write_progress(**kw):
    try:
        cur = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
        cur.update(kw); cur["updated_at"] = time.time()
        PROGRESS.write_text(json.dumps(cur), encoding="utf-8")
    except Exception:
        pass


def load_model(adapter_dir, page_timeout):
    import torch
    try:
        from transformers.utils.import_utils import is_torch_fx_available  # noqa
    except ImportError:
        import transformers.utils.import_utils as _i, transformers.utils as _t
        _i.is_torch_fx_available = _t.is_torch_fx_available = lambda: False
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                     use_safetensors=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()      # fold LoRA into base -> normal infer()/type
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    # per-page wall-clock cap (same safety as the base gen)
    pt = float(page_timeout)
    try:
        model.generation_config.max_time = pt
    except Exception:
        pass
    _g = model.generate
    def _timed(*a, **k):
        k.setdefault("max_time", pt); return _g(*a, **k)
    model.generate = _timed
    return model, tok


def run_one(model, tok, img):
    tmp = Path(tempfile.mkdtemp())
    try:
        t1 = time.time()
        # max_length 2048 (NOT 8192): the FT model emits a complete document (~500
        # tokens) then EOS at 2048; at 8192 it rambles past \end{document} and the
        # no-repeat processor crawls, hitting the timeout with garbage.
        model.infer(tok, prompt=FT_PROMPT, image_file=str(img), output_path=str(tmp),
                    base_size=1024, image_size=640, crop_mode=True, max_length=2048,
                    no_repeat_ngram_size=35, ngram_window=128, save_results=True)
        dt = time.time() - t1
        md = tmp / "result.md"
        raw = md.read_text(encoding="utf-8", errors="replace") if md.exists() else ""
        return raw, dt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=700)
    ap.add_argument("--adapter", default="")  # blank -> auto (best/ else final/ else root)
    ap.add_argument("--page-timeout", type=float, default=120.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    adapter = Path(args.adapter) if args.adapter else None
    if adapter is None:
        for c in (DEFAULT_ADAPTER / "best", DEFAULT_ADAPTER / "final", DEFAULT_ADAPTER):
            if (c / "adapter_config.json").exists():
                adapter = c; break
    if adapter is None or not (adapter / "adapter_config.json").exists():
        print(f"ERROR: no LoRA adapter found under {DEFAULT_ADAPTER}", flush=True)
        write_progress(status="error", error="adapter not found")
        sys.exit(2)
    print(f"using adapter: {adapter}", flush=True)

    rows = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()][:args.samples]

    def done(r):
        f = OUT / f"{r['id']}.tex"
        return f.exists() and f.stat().st_size >= 50

    pending = [r for r in rows if not done(r)]
    total, existing = len(rows), len(rows) - len(pending)
    print(f"Baidu FT: {total} target, {len(pending)} to run", flush=True)
    write_progress(status="loading", model="Baidu OCR FT", total=total, done=existing,
                   started_at=time.time(), adapter=str(adapter))

    print("loading base + adapter...", flush=True)
    t0 = time.time()
    model, tok = load_model(adapter, args.page_timeout)
    print(f"model ready in {time.time()-t0:.0f}s", flush=True)
    write_progress(status="running")

    times = []
    for i, r in enumerate(pending):
        pid = r["id"]
        try:
            raw, dt = run_one(model, tok, IMG / f"{pid}.png")
            tex = ft_format(raw)
        except Exception as e:
            print(f"  {pid} FAIL: {e}", flush=True)
            raw, tex, dt = "", ft_format(""), 0.0
        times.append(dt)
        (OUT / f"{pid}.tex").write_text(tex, encoding="utf-8")
        (OUT / f"{pid}.raw").write_text(raw, encoding="utf-8")
        (OUT / f"{pid}.time").write_text(str(round(dt, 2)), encoding="utf-8")
        done_n = existing + i + 1
        eta = (sum(times) / len(times)) * (len(pending) - i - 1) / 3600
        print(f"  [{done_n}/{total}] {pid}: {len(tex)} chars, {dt:.1f}s, ETA {eta:.2f}h", flush=True)
        write_progress(status="running", done=done_n, total=total, current=pid,
                       pct=round(100 * done_n / total, 1), eta_h=round(eta, 2),
                       avg_s=round(sum(times) / len(times), 1))

    write_progress(status="done", done=total, total=total, current="", eta_h=0.0)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
