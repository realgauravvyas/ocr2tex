"""Classify why validate rejected files: real truncation vs heuristic false positives."""
import sys
import re
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, r"D:\ocr2tex\dashboard")
from app import looks_truncated, compile_latex_check

REJ = Path(r"D:\ocr2tex\workspace\7_rejected")


def main():
    rej = sorted(REJ.glob("*.tex"))
    print("rejected files:", len(rej))
    reasons = Counter()
    trunc_files = []
    for f in rej:
        t = f.read_text(encoding="utf-8", errors="replace")
        if not looks_truncated(t):
            reasons["compile_fail"] += 1
            continue
        trunc_files.append(f)
        body = t
        m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", t, re.DOTALL)
        if m:
            body = m.group(1)
        s = body.rstrip()
        last = s.splitlines()[-1].strip() if s.splitlines() else ""
        dollars = len(re.findall(r"(?<!\\)\$", body))
        if dollars % 2 != 0:
            reasons["unbalanced_dollars"] += 1
        elif body.count(r"\[") != body.count(r"\]"):
            reasons["unbalanced_brackets"] += 1
        elif len(re.findall(r"\\begin\{", body)) != len(re.findall(r"\\end\{", body)):
            reasons["unbalanced_envs"] += 1
        elif re.search(r"\\\\\s*$", last):
            reasons["ends_with_double_backslash"] += 1
        else:
            reasons["dangling_operator"] += 1
    print(dict(reasons))

    random.seed(1)
    sample = random.sample(trunc_files, min(20, len(trunc_files)))
    ok = 0
    for f in sample:
        good, _ = compile_latex_check(f.read_text(encoding="utf-8", errors="replace"))
        ok += good
    print(f"sampled {len(sample)} TRUNCATED-flagged files: {ok} actually compile fine")


if __name__ == "__main__":
    main()
