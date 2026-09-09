r"""v3 output formatter.

BUG 6 (pure post-processing, costs no GPU time to fix):

  baidu_format.ft_format() returns `raw` unchanged only when BOTH \documentclass
  and \end{document} are present; otherwise it wraps `raw` in PREAMBLE+POSTAMBLE.
  A truncated generation still contains \documentclass ... \begin{document}, so
  the wrap produces a document with TWO \documentclass and TWO \begin{document}
  but ONE \end{document}:

      \documentclass{article} ... \begin{document}      <- injected preamble
      \documentclass{article} ... \begin{document}      <- model's own output
      ...body...
      \end{document}                                    <- injected postamble

  That is never compilable and always fails struct_ok's
  count("\\begin") == count("\\end") check. It hit 123 of the 700 v1 test pages
  (17.6%) -- consistent with struct_pct 71.7 / compile_rate 62.7 vs GLM-OCR
  v4.1's 90.9 / 82.4.

v3 repairs instead of wrapping: keep the model's own preamble, drop a degenerate
repeated tail, close whatever math/environments/braces are still open, then add
the single missing \end{document}.
"""
import re

PREAMBLE = ("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n"
            "\\usepackage{amsfonts}\n\n\\begin{document}\n\n")
POSTAMBLE = "\n\n\\end{document}\n"

_DET = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
_SPECIAL = re.compile(r"<\|/?[a-z_]+\|>")
_IMGPLACEHOLDER = re.compile(r"!\[\]\(images/\d+\.jpg\)")
_BEGIN = re.compile(r"\\begin\s*\{([^}]*)\}")
_END = re.compile(r"\\end\s*\{([^}]*)\}")


def strip_special(raw: str) -> str:
    raw = (raw or "")
    raw = _DET.sub("", raw)
    raw = _SPECIAL.sub("", raw)
    raw = _IMGPLACEHOLDER.sub("", raw)
    return raw.strip()


# --------------------------------------------------------------------------- #
# degenerate-tail trimming
# --------------------------------------------------------------------------- #
_DIGITS = re.compile(r"\d+")


def _loop_key(tok: str, normalize_digits: bool) -> str:
    """Comparison key for periodicity. The observed decode loops are NOT byte
    identical -- they carry an incrementing counter, e.g.
        $\\in I_{36}$ \\quad $\\in I_{37}$ \\quad $\\in I_{38}$ ...
    so an exact-equality period test never fires and the loop survives (measured:
    3/24 eval pages at len_rate 3.3-4.0 passed straight through). Collapsing digit
    runs makes those blocks compare equal while leaving non-numeric text alone.
    """
    return _DIGITS.sub("#", tok) if normalize_digits else tok


def trim_degenerate_tail(text: str, min_reps: int = 4, max_period: int = 60,
                         window: int = 600, normalize_digits: bool = True) -> str:
    """Cut a decode loop off the end of the text, conservatively.

    Two passes, both anchored at the END (a repetition in the middle of a page is
    usually real content -- e.g. an enumerated list -- so we never touch it):
      1. line level: a run of >=3 identical trailing lines collapses to one
         (digit-insensitive when normalize_digits, to catch numbered loops).
      2. token level: if the last `window` whitespace tokens end in a block that
         is p-periodic for >= min_reps periods, keep two periods.
    Returns the text unchanged when no loop is detected.
    """
    lines = text.rstrip().split("\n")
    def lk(s):
        return _loop_key(s, normalize_digits)
    while (len(lines) >= 3 and lines[-1].strip()
           and lk(lines[-1]) == lk(lines[-2]) == lk(lines[-3])):
        lines.pop()
    text = "\n".join(lines)

    toks = text.split()
    if len(toks) < 40:
        return text
    tail = toks[-window:]
    keys = [_loop_key(t, normalize_digits) for t in tail]
    for p in range(1, min(max_period, len(tail) // min_reps) + 1):
        unit = keys[-p:]
        reps = 1
        while (reps + 1) * p <= len(tail) and keys[-(reps + 1) * p:-reps * p] == unit:
            reps += 1
        if reps >= min_reps:
            drop = (reps - 2) * p                      # keep two periods
            if drop > 0:
                keep = toks[:len(toks) - drop]
                # rejoin on the original text so we keep newlines up to the cut
                idx, cnt = 0, 0
                for m in re.finditer(r"\S+", text):
                    cnt += 1
                    if cnt == len(keep):
                        idx = m.end()
                        break
                return text[:idx] if idx else " ".join(keep)
            return text
    return text


# --------------------------------------------------------------------------- #
# LaTeX balancing
# --------------------------------------------------------------------------- #
def _scan(text):
    """Walk the source tracking escapes, comments, $-math, brace depth and the
    \\begin/\\end environment stack. Returns (brace_depth, dollar_open,
    dollardollar_open, env_stack)."""
    depth, dollar, ddollar, envs = 0, False, False, []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            m = _BEGIN.match(text, i)
            if m:
                envs.append(m.group(1))
                i = m.end()
                continue
            m = _END.match(text, i)
            if m:
                if envs and envs[-1] == m.group(1):
                    envs.pop()
                elif m.group(1) in envs:
                    while envs and envs.pop() != m.group(1):
                        pass
                i = m.end()
                continue
            i += 2                                     # escaped char, incl. \{ \} \$ \%
            continue
        if c == "%":
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == "$":
            if text.startswith("$$", i):
                ddollar = not ddollar
                i += 2
                continue
            dollar = not dollar
        elif c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        i += 1
    return depth, dollar, ddollar, envs


_TRAILING_FRAGMENT = re.compile(r"(\\[a-zA-Z]*|\\)$")


def close_open_constructs(body: str) -> str:
    """Close whatever the truncated generation left open, innermost first."""
    body = body.rstrip()
    body = _TRAILING_FRAGMENT.sub("", body).rstrip()    # half-typed \fra / bare \
    depth, dollar, ddollar, envs = _scan(body)
    out = [body]
    if ddollar:
        out.append("$$")
    if dollar:
        out.append("$")
    for env in reversed(envs):
        if env == "document":
            continue
        out.append("\n\\end{%s}" % env)
    if depth > 0:
        out.append("}" * depth)
    return "".join(out)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def ft_format_v3(raw: str, repair: bool = True, trim_repeats: bool = True,
                 normalize_digits: bool = True) -> str:
    """Turn one raw generation into a scoreable LaTeX document.

    Never double-wraps: if the model emitted its own preamble we keep exactly
    that one.
    """
    text = strip_special(raw)
    if not text:
        return PREAMBLE + POSTAMBLE

    dc = text.find("\\documentclass")
    if dc > 0:
        text = text[dc:]                               # drop chatter before the doc

    end = text.find("\\end{document}")
    if end != -1:                                      # complete -- drop trailing junk
        return text[:end + len("\\end{document}")].rstrip() + "\n"

    if not repair:                                     # v1/v2 behaviour, for A/B only
        body = re.sub(r"^#{1,6}\s*", "", text, flags=re.M).strip()
        return PREAMBLE + body + POSTAMBLE

    if trim_repeats:
        text = trim_degenerate_tail(text, normalize_digits=normalize_digits)

    if dc == -1 and "\\documentclass" not in text:     # model never wrote a preamble
        body = re.sub(r"^#{1,6}\s*", "", text, flags=re.M).strip()
        return PREAMBLE + close_open_constructs(body) + POSTAMBLE

    if "\\begin{document}" not in text:                # cut off inside the preamble
        text = text.rstrip() + "\n\n\\begin{document}\n"

    return close_open_constructs(text) + POSTAMBLE
