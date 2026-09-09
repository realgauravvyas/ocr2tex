"""Metric access for v3.

Imports the EXISTING scorer (D:\\ocr2tex\\dashboard\\benchmark_glm_ocr.py) read-only
so v3 numbers are directly comparable with the recorded Base GLM-OCR /
glm-ocr-math-v4.1 / Baidu FT v1 / v2 columns. Nothing is written back there.

If that import fails (missing numpy etc. in whichever interpreter is used), a
reduced fallback provides CER / Norm-CER / struct so a probe can still run --
the fallback is flagged in the output so nobody mistakes it for the full suite.
"""
import sys
from pathlib import Path

DASH = Path(r"D:\ocr2tex\dashboard")
FULL = True

try:
    if str(DASH) not in sys.path:
        sys.path.insert(0, str(DASH))
    from benchmark_glm_ocr import score, summarize, compile_tex, struct_ok  # noqa: F401
except Exception as _e:                                    # pragma: no cover
    FULL = False
    _IMPORT_ERROR = _e
    import re

    def _norm(s):
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _lev(a, b):
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def struct_ok(text):
        return ("\\documentclass" in text and "\\end{document}" in text
                and text.count("{") == text.count("}")
                and text.count("\\begin") == text.count("\\end"))

    def _canon(s):
        m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", s, re.DOTALL)
        if m:
            s = m.group(1)
        s = re.sub(r"\\[dt]frac", r"\\frac", s)
        s = re.sub(r"\\(?:left|right|big{1,2}[lr]?)\b", "", s)
        s = s.replace("$", "")
        return re.sub(r"\s+", "", s)

    def score(ref, hyp):
        nref, nhyp = _norm(ref), _norm(hyp)
        cer = _lev(nref, nhyp) / max(1, len(nref))
        cref, chyp = _canon(ref), _canon(hyp)
        return {"cer": round(cer, 4),
                "ncer": round(_lev(cref, chyp) / max(1, len(cref)), 4),
                "charsim": round(max(0.0, 1 - cer), 4),
                "struct": struct_ok(hyp),
                "len_rate": round(len(nhyp) / max(1, len(nref)), 3)}

    def summarize(results):
        if not results:
            return {}
        n = len(results)
        keys = ["cer", "ncer", "charsim", "len_rate"]
        out = {"samples": n, "PARTIAL_METRICS": True}
        for k in keys:
            out["mean_" + k if k == "cer" else k] = round(sum(r[k] for r in results) / n, 4)
        out["struct_pct"] = round(100.0 * sum(r["struct"] for r in results) / n, 1)
        return out

    def compile_tex(tex_str, out_pdf=None, timeout=30):
        return None, None


def reason():
    return "full benchmark_glm_ocr metrics" if FULL else f"FALLBACK metrics ({_IMPORT_ERROR})"
