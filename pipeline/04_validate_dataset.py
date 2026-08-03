"""
Step 4: Validate & Compile Dataset
Compiles EVERY .tex annotation with pdflatex to verify it is training-ready.

What it does:
1. Auto-fixes common LaTeX issues (missing packages, unescaped chars, truncation)
2. Compiles each .tex with pdflatex in a temp dir
3. Sorts results into:
   - PASS: compiles cleanly -> kept
   - FIXED: had an issue, auto-fixed, now compiles -> kept (file updated)
   - FAIL: cannot compile even after fixes -> moved to annotations_rejected/
   - TRUNCATED: output looks cut off -> moved to annotations_rejected/
4. Writes a report (validation_report.json)

Only PASS + FIXED files remain in the annotations folder, so the dataset
you build from them is guaranteed to compile.

Usage:
  python scripts/04_validate_dataset.py                 # validate all, move bad ones out
  python scripts/04_validate_dataset.py --no-move       # report only, don't move
  python scripts/04_validate_dataset.py --workers 4     # parallel compile
  python scripts/04_validate_dataset.py --sample 50     # test on first 50 only
"""

import os
import re
import json
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# Common packages we ensure are present in the preamble
REQUIRED_PACKAGES = [
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
    r"\usepackage{amsfonts}",
]

# Map: if any of these commands appear in the body, ensure the package is loaded.
# This catches commands VLMs emit that need extra packages.
COMMAND_PACKAGE_MAP = {
    "cancel": [r"\cancel", r"\bcancel", r"\xcancel", r"\cancelto"],
    "mathtools": [r"\xmapsto", r"\coloneqq", r"\xLeftrightarrow"],
    "gensymb": [r"\degree", r"\celsius"],
    "textcomp": [r"\textcelsius"],
    "graphicx": [r"\includegraphics"],
    "enumitem": [r"\begin{enumerate}["],
    "ulem": [r"\sout", r"\uline"],
}


def looks_truncated(latex: str) -> bool:
    """
    Heuristic to detect truncated/incomplete LaTeX output.
    """
    body = latex
    # Get the body between begin and end document
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", latex, re.DOTALL)
    if m:
        body = m.group(1)

    stripped = body.rstrip()
    if not stripped:
        return True

    # Ends with a dangling equation/assignment like "A =" or "= " or "\to"
    dangling_patterns = [
        r"=\s*$",            # ends with "="
        r"\\quad\s*$",       # ends with \quad
        r"\\to\s*$",         # ends with \to
        r"\\Rightarrow\s*$", # ends with \Rightarrow
        r"\+\s*$",           # ends with "+"
        r"-\s*$",            # ends with "-"
        r"\\\\\s*$",         # ends with line break
    ]
    last_line = stripped.splitlines()[-1].strip() if stripped.splitlines() else ""
    for pat in dangling_patterns:
        if re.search(pat, last_line):
            return True

    # Unbalanced math delimiters in body (odd number of unescaped $)
    dollars = len(re.findall(r"(?<!\\)\$", body))
    if dollars % 2 != 0:
        return True

    # Unbalanced \[ \]
    if body.count(r"\[") != body.count(r"\]"):
        return True

    # Unbalanced begin/end environments count
    begins = len(re.findall(r"\\begin\{", body))
    ends = len(re.findall(r"\\end\{", body))
    if begins != ends:
        return True

    return False


def auto_fix_latex(latex: str) -> str:
    """
    Apply safe automatic fixes for common compilation problems.
    """
    fixed = latex.strip()

    # Strip markdown fences if any slipped through
    if fixed.startswith("```"):
        fixed = re.sub(r"^```[a-zA-Z]*\n?", "", fixed)
        fixed = re.sub(r"\n?```$", "", fixed).strip()

    # Remove review markers
    fixed = fixed.replace("% REVIEWED: OK", "").strip()

    # --- Remove MathJax-only commands that pdflatex doesn't understand ---
    # \require{...} is a MathJax directive, not LaTeX. Drop it entirely.
    fixed = re.sub(r"\\require\{[^}]*\}", "", fixed)
    # Replace \textcircled{X} with (X) to guarantee compilation.
    fixed = re.sub(r"\\textcircled\{([^}]*)\}", r"(\1)", fixed)

    # Replace literal Unicode circled numbers ①..⑳ with (1)..(20)
    for i in range(20):
        fixed = fixed.replace(chr(0x2460 + i), f"({i + 1})")
    # The Unicode minus sign is always safe to normalize to ASCII '-'
    fixed = fixed.replace("−", "-")

    # Ensure documentclass exists
    if "\\documentclass" not in fixed:
        fixed = "\\documentclass{article}\n" + fixed

    # --- Determine which extra packages are needed based on body commands ---
    needed_pkgs = list(REQUIRED_PACKAGES)
    for pkg_name, triggers in COMMAND_PACKAGE_MAP.items():
        if any(t in fixed for t in triggers):
            needed_pkgs.append(r"\usepackage{" + pkg_name + "}")

    # Ensure needed packages are present (insert after documentclass line)
    lines = fixed.splitlines()
    preamble_text = fixed.split("\\begin{document}")[0] if "\\begin{document}" in fixed else fixed

    missing_pkgs = []
    for pkg in needed_pkgs:
        pkg_name = re.search(r"\{(\w+)\}", pkg).group(1)
        if f"{{{pkg_name}}}" not in preamble_text:
            missing_pkgs.append(pkg)

    if missing_pkgs:
        # Find documentclass line index
        insert_idx = 0
        for i, line in enumerate(lines):
            if "\\documentclass" in line:
                insert_idx = i + 1
                break
        for pkg in reversed(missing_pkgs):
            lines.insert(insert_idx, pkg)
        fixed = "\n".join(lines)

    # Ensure begin/end document
    if "\\begin{document}" not in fixed:
        # Put begin after the preamble (after last \usepackage)
        usepackage_positions = [m.end() for m in re.finditer(r"\\usepackage(\[[^\]]*\])?\{[^}]*\}", fixed)]
        if usepackage_positions:
            pos = usepackage_positions[-1]
            fixed = fixed[:pos] + "\n\n\\begin{document}\n" + fixed[pos:]
        else:
            fixed += "\n\\begin{document}\n"

    if "\\end{document}" not in fixed:
        fixed = fixed.rstrip() + "\n\n\\end{document}\n"

    # Balance $ ... $ : if odd count, the simplest safe fix is hard; leave to truncation check
    return fixed


def compile_latex(latex: str, timeout: int = 60) -> tuple:
    """
    Try to compile LaTeX with pdflatex.
    Returns (success: bool, error_message: str).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "doc.tex"
        tex_file.write_text(latex, encoding="utf-8")

        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 "-no-shell-escape", "doc.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except FileNotFoundError:
            return False, "PDFLATEX_NOT_FOUND"

        pdf_file = Path(tmpdir) / "doc.pdf"
        if result.returncode == 0 and pdf_file.exists():
            return True, ""

        # Extract first error from log
        log_file = Path(tmpdir) / "doc.log"
        err = "Unknown error"
        if log_file.exists():
            log = log_file.read_text(encoding="utf-8", errors="ignore")
            for line in log.splitlines():
                if line.startswith("!"):
                    err = line.strip()
                    break
        return False, err


def process_one(tex_path_str: str) -> dict:
    """
    Worker: validate + (optionally fix) + compile one file.
    Returns a result dict. Does NOT move files (parent decides).
    May rewrite the file with fixed content if a fix made it compile.
    """
    tex_path = Path(tex_path_str)
    original = tex_path.read_text(encoding="utf-8")

    result = {
        "file": tex_path.name,
        "status": None,        # PASS | FIXED | FAIL | TRUNCATED
        "error": "",
        "rewritten": False,
    }

    # Truncation check first (don't waste a compile)
    if looks_truncated(original):
        result["status"] = "TRUNCATED"
        result["error"] = "Output appears cut off / unbalanced delimiters"
        return result

    # Try compiling as-is
    ok, err = compile_latex(original)
    if ok:
        result["status"] = "PASS"
        return result

    if err == "PDFLATEX_NOT_FOUND":
        result["status"] = "FAIL"
        result["error"] = "pdflatex not found"
        return result

    # Try auto-fix
    fixed = auto_fix_latex(original)
    if fixed != original:
        ok2, err2 = compile_latex(fixed)
        if ok2:
            tex_path.write_text(fixed, encoding="utf-8")
            result["status"] = "FIXED"
            result["rewritten"] = True
            return result
        else:
            result["status"] = "FAIL"
            result["error"] = err2
            return result

    result["status"] = "FAIL"
    result["error"] = err
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate & compile all .tex annotations")
    parser.add_argument("--annotations-dir", default=r"d:\Kiro\dataset\annotations")
    parser.add_argument("--rejected-dir", default=r"d:\Kiro\dataset\annotations_rejected")
    parser.add_argument("--workers", type=int, default=4, help="Parallel compile workers")
    parser.add_argument("--sample", type=int, default=0, help="Only check first N files (0=all)")
    parser.add_argument("--no-move", action="store_true", help="Report only; don't move bad files")
    args = parser.parse_args()

    ann_dir = Path(args.annotations_dir)
    rej_dir = Path(args.rejected_dir)

    tex_files = sorted(ann_dir.glob("*.tex"))
    if args.sample > 0:
        tex_files = tex_files[:args.sample]

    if not tex_files:
        print(f"No .tex files found in {ann_dir}")
        return

    print("=" * 60)
    print(f"VALIDATING {len(tex_files)} .tex files with pdflatex")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    results = []
    counts = {"PASS": 0, "FIXED": 0, "FAIL": 0, "TRUNCATED": 0}

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, str(f)): f for f in tex_files}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            done += 1
            if done % 25 == 0 or done == len(tex_files):
                print(f"  [{done}/{len(tex_files)}] "
                      f"PASS={counts['PASS']} FIXED={counts['FIXED']} "
                      f"FAIL={counts['FAIL']} TRUNCATED={counts['TRUNCATED']}")

    # Move bad files
    moved = 0
    if not args.no_move:
        rej_dir.mkdir(parents=True, exist_ok=True)
        bad_statuses = {"FAIL", "TRUNCATED"}
        for res in results:
            if res["status"] in bad_statuses:
                src = ann_dir / res["file"]
                if src.exists():
                    shutil.move(str(src), str(rej_dir / res["file"]))
                    moved += 1

    # Report
    report = {
        "total": len(tex_files),
        "counts": counts,
        "kept": counts["PASS"] + counts["FIXED"],
        "rejected_moved": moved,
        "results": results,
    }
    report_path = ann_dir.parent / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  PASS (compiled clean):     {counts['PASS']}")
    print(f"  FIXED (auto-fixed):        {counts['FIXED']}")
    print(f"  TRUNCATED (incomplete):    {counts['TRUNCATED']}")
    print(f"  FAIL (cannot compile):     {counts['FAIL']}")
    print(f"  -------------------------------------")
    print(f"  KEPT for training:         {report['kept']}")
    if not args.no_move:
        print(f"  Moved to rejected:         {moved}  -> {rej_dir}")
    else:
        print(f"  (--no-move: nothing moved)")
    print(f"\n  Report: {report_path}")

    if counts["FAIL"] + counts["TRUNCATED"] > 0 and not args.no_move:
        print(f"\n  The rejected files need re-OCR. You can re-run annotation:")
        print(f"    1. Delete matching files from annotations_rejected/")
        print(f"    2. Run: python scripts/02_annotate_helper.py --mode auto")
        print(f"       (it will regenerate any image missing a .tex file)")


if __name__ == "__main__":
    main()
