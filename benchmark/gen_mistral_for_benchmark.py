"""Run Mistral OCR (mistral-ocr-latest / "OCR 4") on the GLM-OCR test set via the
Mistral API and save per-page outputs, reformatted into the SAME GLM-OCR LaTeX
shape as the other models so the metrics are comparable.

Mistral OCR is an API service (NOT downloadable open weights), so this calls the
REST endpoint per page. No GPU — requests run in parallel, so 700 pages take
minutes and it can run alongside the local Baidu GPU benchmark.

Cost: ~$4 / 1000 pages (so ~$2.80 for 700). Needs a Mistral API key:
  set MISTRAL_API_KEY=...   (or pass --api-key)
"""
import os, sys, json, time, base64, argparse, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"D:\ocr2tex\dashboard")
from baidu_format import to_glm_format  # same GLM-OCR envelope + furniture stripping

TEST = Path(r"D:\ocr2tex\workspace\9_split\test.jsonl")
IMG = Path(r"D:\ocr2tex\workspace\9_split\images")
OUT = Path(r"D:\ocr2tex\output\bench_outputs\Mistral_OCR_4")
PROGRESS = Path(r"D:\ocr2tex\output\mistral_gen_progress.json")
ENDPOINT = "https://api.mistral.ai/v1/ocr"
MODEL = "mistral-ocr-latest"


def write_progress(**kw):
    try:
        cur = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
        cur.update(kw); cur["updated_at"] = time.time()
        PROGRESS.write_text(json.dumps(cur), encoding="utf-8")
    except Exception:
        pass


def ocr_one(pid, api_key, timeout=120, retries=4):
    """Call Mistral OCR on one page; return the page markdown (raw)."""
    img_b64 = base64.b64encode((IMG / f"{pid}.png").read_bytes()).decode()
    body = json.dumps({
        "model": MODEL,
        "document": {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
        "include_image_base64": False,
    }).encode()
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
                "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            pages = data.get("pages") or []
            return "\n\n".join(p.get("markdown", "") for p in pages).strip()
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503, 504):  # rate-limit / transient → backoff
                time.sleep(2 ** attempt + 1); continue
            raise
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(2 ** attempt + 1)
    raise RuntimeError(f"Mistral OCR failed for {pid}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=700)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--api-key", default=os.environ.get("MISTRAL_API_KEY", ""))
    args = ap.parse_args()
    if not args.api_key:
        print("ERROR: no Mistral API key (set MISTRAL_API_KEY or pass --api-key)", flush=True)
        sys.exit(2)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()][:args.samples]

    def done(r):
        f = OUT / f"{r['id']}.tex"
        return f.exists() and f.stat().st_size >= 50

    pending = [r for r in rows if not done(r)]
    total, existing = len(rows), len(rows) - len(pending)
    print(f"Mistral OCR: {total} target, {len(pending)} to run, {args.workers} workers", flush=True)
    write_progress(status="running", model=MODEL, total=total, done=existing,
                   started_at=time.time(), recent=[])

    t0 = [time.time()]
    counter = [existing]

    def work(r):
        pid = r["id"]
        t1 = time.time()
        try:
            raw = ocr_one(pid, args.api_key)
        except Exception as e:
            print(f"  {pid} FAIL: {e}", flush=True)
            raw = ""
        dt = time.time() - t1
        tex = to_glm_format(raw)
        (OUT / f"{pid}.tex").write_text(tex, encoding="utf-8")
        (OUT / f"{pid}.raw").write_text(raw, encoding="utf-8")
        (OUT / f"{pid}.time").write_text(str(round(dt, 2)), encoding="utf-8")
        counter[0] += 1
        n = counter[0]
        rate = (time.time() - t0[0]) / max(1, n - existing)
        eta = rate * (total - n) / 3600
        print(f"  [{n}/{total}] {pid}: {len(tex)} chars, {dt:.1f}s, ETA {eta:.2f}h", flush=True)
        write_progress(status="running", done=n, total=total, current=pid,
                       pct=round(100 * n / total, 1), eta_h=round(eta, 2))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, pending))

    write_progress(status="done", done=total, total=total, current="", eta_h=0.0)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
