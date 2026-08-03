"""Requeue genuinely bad rejected annotations for re-annotation.

Files in 7_rejected that the FIXED truncation heuristic still considers broken,
plus real compile failures, get their 6_annotations/*.tex deleted so the next
annotate pass regenerates them. False positives of the old \\[ counting bug are
kept — the fixed validator will rescue them on the next pass.
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, r"D:\ocr2tex\dashboard")
from app import looks_truncated  # fixed version

ANN = Path(r"D:\ocr2tex\workspace\6_annotations")
REJ = Path(r"D:\ocr2tex\workspace\7_rejected")


def old_looks_truncated(latex):
    body = latex
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", latex, re.DOTALL)
    if m:
        body = m.group(1)
    stripped = body.rstrip()
    if not stripped:
        return True
    dangling = [r"=\s*$", r"\\quad\s*$", r"\\to\s*$", r"\\Rightarrow\s*$", r"\+\s*$", r"-\s*$", r"\\\\\s*$"]
    last_line = stripped.splitlines()[-1].strip() if stripped.splitlines() else ""
    for pat in dangling:
        if re.search(pat, last_line):
            return True
    dollars = len(re.findall(r"(?<!\\)\$", body))
    if dollars % 2 != 0:
        return True
    if body.count(r"\[") != body.count(r"\]"):
        return True
    begins = len(re.findall(r"\\begin\{", body))
    ends = len(re.findall(r"\\end\{", body))
    if begins != ends:
        return True
    return False


def main():
    rescued = requeued = missing = 0
    for f in sorted(REJ.glob("*.tex")):
        t = f.read_text(encoding="utf-8", errors="replace")
        if old_looks_truncated(t) and not looks_truncated(t):
            rescued += 1
            continue
        src = ANN / f.name
        if src.exists():
            src.unlink()
            requeued += 1
        else:
            missing += 1
    print(f"rescued (kept, will revalidate): {rescued}")
    print(f"requeued for re-annotation: {requeued}")
    print(f"missing in 6_annotations: {missing}")


if __name__ == "__main__":
    main()
