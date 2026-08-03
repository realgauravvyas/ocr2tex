"""Canonical formatter that turns Unlimited-OCR's raw det-tagged output into a
GLM-OCR-style LaTeX document (handwritten content only), so the two models are
comparable. Single source of truth — the live scorer re-derives every .tex from
the saved .raw stream through here, so furniture rules can be refined without
re-running the 7.5h GPU generation.
"""
import re

PREAMBLE = ("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n"
            "\\usepackage{amsfonts}\n\n\\begin{document}\n\n")
POSTAMBLE = "\n\n\\end{document}\n"

# Region types that are printed page furniture (v4.1 was trained to omit these).
DROP_TYPES = {"header", "footer", "page_number", "page-number", "pagenumber"}
DET_RE = re.compile(r"<\|det\|>\s*([a-z_\-]+)\s*\[[0-9,\s]*\]\s*<\|/det\|>", re.I)

# Printed strings/markers that leak when the model mis-types them as `text`.
# The KV exam pages carry a fixed set of printed instructions + question markers.
FURNITURE_RE = re.compile(
    r"(Space\s+for\s+answering"
    r"|Extra\s+space\s+for\s+answer"
    r"|not\s+for\s+rough\s+work"
    r"|Mention\s+Question\s+Number"
    r"|^\s*\d+\s+of\s+\d+\s*$"                          # "5 of 20" page number
    r"|^\s*Q\s*[-.\)①-⑳\d\?\s]{0,10}$)",      # bare question marker, e.g. "Q-3)", "Q-? ③ 2"
    re.I,
)


def strip_tokens(t: str) -> str:
    t = re.sub(r"<\|det\|>.*?<\|/det\|>", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|/?[a-z_]+\|>", "", t)
    t = re.sub(r"!\[\]\(images/\d+\.jpg\)", "", t)      # image placeholders
    return t


def _drop_furniture_lines(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not FURNITURE_RE.search(ln.strip()))


def ft_format(raw: str) -> str:
    """Formatter for the FINE-TUNED Baidu model, which was trained to emit a full
    GLM-style LaTeX document directly. So use the output as-is (just strip stray
    special tokens); only wrap if the model didn't produce a complete document."""
    raw = (raw or "").strip()
    raw = re.sub(r"<\|det\|>.*?<\|/det\|>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<\|/?[a-z_]+\|>", "", raw).strip()
    if "\\documentclass" in raw and "\\end{document}" in raw:
        return raw  # already a complete document — don't double-wrap
    body = re.sub(r"^#{1,6}\s*", "", raw, flags=re.M).strip()
    return PREAMBLE + body + POSTAMBLE


def to_glm_format(raw: str) -> str:
    raw = (raw or "").strip()
    parts = DET_RE.split(raw)   # [pre, type1, content1, type2, content2, ...]
    if len(parts) >= 3:
        chunks = []
        for i in range(1, len(parts), 2):
            rtype = parts[i].lower()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            if rtype in DROP_TYPES:
                continue
            content = _drop_furniture_lines(strip_tokens(content)).strip()
            if content:
                chunks.append(content)
        body = "\n\n".join(chunks)
    else:
        body = _drop_furniture_lines(strip_tokens(raw))
    body = re.sub(r"^#{1,6}\s*", "", body, flags=re.M)  # markdown headers
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return PREAMBLE + body + POSTAMBLE
