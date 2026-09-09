r"""Score a v3 bench folder and print it next to the already-recorded columns.

Reads D:\ocr2tex\output\benchmark_results.json READ-ONLY purely to print the
existing Base GLM-OCR / glm-ocr-math-v4.1 / Baidu FT v1 / v2 numbers in the same
table. All v3 output goes to bench/<label>_scores.json inside this folder --
benchmark_results.json is never modified, so the dashboard is unaffected.

Every .tex is re-derived from the saved .raw through format_v3, so formatter
changes can be re-scored without touching the GPU.
"""
import os, sys, json, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

V3 = Path(__file__).resolve().parent
sys.path.insert(0, str(V3))

from common_v3 import read_jsonl, target_of, DATA_DIR, write_json_atomic
from format_v3 import ft_format_v3
import metrics_v3

BENCH = V3 / "bench"
EXISTING = Path(r"D:\ocr2tex\output\benchmark_results.json")   # READ-ONLY

COLS = [("mean_cer", "CER", 4, False), ("norm_cer", "nCER", 4, False),
        ("bleu", "BLEU", 4, True), ("chrf", "chrF", 4, True),
        ("math_f1", "MathF1", 4, True), ("struct_pct", "Struct%", 1, True),
        ("compile_rate", "Comp%", 1, True), ("cer_under_30pct", "CER<30%", 1, True),
        ("len_rate", "Len", 3, True), ("avg_latency_s", "sec", 1, False)]


def score_dir(label, refs, workers, no_compile, repair, trim):
    d = BENCH / label
    if not d.exists():
        sys.exit(f"ERROR: {d} does not exist - run gen_baidu_v3.py --label {label} first")
    pids = sorted(p.stem for p in d.glob("*.raw"))
    print(f"scoring {label}: {len(pids)} pages ({metrics_v3.reason()})", flush=True)

    def one(pid):
        raw = (d / f"{pid}.raw").read_text(encoding="utf-8", errors="replace")
        hyp = ft_format_v3(raw, repair=repair, trim_repeats=trim)
        (d / f"{pid}.tex").write_text(hyp, encoding="utf-8")
        row = metrics_v3.score(refs.get(pid, ""), hyp)
        row["id"] = pid
        tf = d / f"{pid}.time"
        try:
            row["latency_s"] = float(tf.read_text().strip()) if tf.exists() else 0.0
        except Exception:
            row["latency_s"] = 0.0
        row["raw_complete"] = "\\end{document}" in raw
        if not no_compile:
            ok, _ = metrics_v3.compile_tex(hyp, d / f"{pid}.pdf")
            row["compile_ok"] = None if ok is None else bool(ok)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, pids))
    s = metrics_v3.summarize(rows)
    s["complete_pct"] = round(100.0 * sum(r["raw_complete"] for r in rows) / max(1, len(rows)), 1)
    return s, rows


def existing_columns():
    if not EXISTING.exists():
        return {}
    try:
        d = json.loads(EXISTING.read_text(encoding="utf-8"))
        return {v["label"]: v.get("summary", {}) for v in d.get("models", {}).values()}
    except Exception:
        return {}


def print_table(rowsets):
    name_w = max(22, max(len(n) for n, _ in rowsets) + 2)
    head = f"{'model':<{name_w}}" + "".join(f"{c[1]:>9}" for c in COLS)
    print("\n" + "=" * len(head))
    print(head)
    print("-" * len(head))
    for name, s in rowsets:
        line = f"{name:<{name_w}}"
        for key, _, prec, _ in COLS:
            v = s.get(key)
            line += f"{'-':>9}" if v is None else f"{v:>9.{prec}f}"
        print(line)
    print("=" * len(head))
    print("CER / nCER / sec: lower is better. Everything else: higher is better.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", dest="labels", default=None,
                    help="repeatable; bench/<label> folders to score (default Baidu_OCR_FT_v3)")
    ap.add_argument("--workers", type=int, default=4, help="pdflatex runs in parallel")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-repair", action="store_true", help="score with the legacy wrap-only formatter")
    ap.add_argument("--no-trim", action="store_true", help="do not trim degenerate tails")
    args = ap.parse_args()

    labels = list(dict.fromkeys(args.labels or ["Baidu_OCR_FT_v3"]))

    rows = read_jsonl(DATA_DIR / "test.jsonl")
    refs = {r["id"]: target_of(r) for r in rows}

    table = []
    prev = existing_columns()
    for name in ["Base GLM-OCR", "glm-ocr-math-v4.1", "Baidu Unlimited-OCR",
                 "Baidu OCR FT", "Baidu OCR FT v2"]:
        if name in prev:
            table.append((name + "  (recorded)", prev[name]))

    for lbl in labels:
        s, per = score_dir(lbl, refs, args.workers, args.no_compile,
                           not args.no_repair, not args.no_trim)
        write_json_atomic(BENCH / f"{lbl}_scores.json",
                          {"label": lbl, "summary": s, "per_sample": per,
                           "repair": not args.no_repair, "trim": not args.no_trim,
                           "metrics": metrics_v3.reason()})
        table.append((lbl, s))

    print_table(table)
    for lbl in labels:
        p = BENCH / f"{lbl}_scores.json"
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
