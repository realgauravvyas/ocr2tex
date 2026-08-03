"""One-off cleanup: strip <think> reasoning blocks from existing annotations.

Rewrites .tex files in workspace/6_annotations in place. Files that contain no
complete LaTeX document after cleaning (reasoning consumed the token budget)
are deleted so the annotate stage re-generates them on resume.
"""
import re
from pathlib import Path

ANN_DIR = Path(r"D:\ocr2tex\workspace\6_annotations")


def strip_reasoning(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"<think>", text, re.IGNORECASE):
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[-1]
    return text


def clean_latex_response(text):
    text = strip_reasoning(text or "").strip()
    m = re.search(r"```(?:latex|tex)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    elif text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("\\documentclass")
    if start != -1:
        end = text.rfind("\\end{document}")
        if end != -1:
            text = text[start : end + len("\\end{document}")]
        else:
            text = text[start:]
    return text.strip()


def main():
    rewritten = deleted = untouched = 0
    for f in sorted(ANN_DIR.glob("*.tex")):
        raw = f.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_latex_response(raw)
        if "\\documentclass" not in cleaned or "\\end{document}" not in cleaned:
            f.unlink()
            deleted += 1
            print(f"DELETED (no complete doc after cleaning): {f.name}")
        elif cleaned != raw.strip():
            f.write_text(cleaned, encoding="utf-8")
            rewritten += 1
        else:
            untouched += 1
    print(f"\nDone: {rewritten} rewritten, {deleted} deleted (will be re-annotated), {untouched} already clean")


if __name__ == "__main__":
    main()
