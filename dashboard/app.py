import os
import sys
import json
import time
import math
import base64
import shutil
import random
import hashlib
import tempfile
import threading
import subprocess
import re
from pathlib import Path
from datetime import datetime
from collections import deque, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from flask import Flask, render_template, jsonify, request, Response, send_file, abort
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
# re-read templates from disk on each request so UI edits take effect on refresh
# without needing a server restart (Jinja caches them in memory otherwise)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


@app.after_request
def add_no_cache(response):
    if response.content_type and 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


BASE_DIR = Path(r"D:\ocr2tex")
DATA_DIR = BASE_DIR / "Data" / "raw data"
WORK_DIR = BASE_DIR / "workspace"

PIPELINE_STAGES = [
    {"id": "crop_pii", "name": "Crop PII (Top 10%)", "icon": "scissors"},
    {"id": "filter_blank", "name": "Filter Blank Pages", "icon": "filter"},
    {"id": "deskew", "name": "Deskew Pages", "icon": "rotate"},
    {"id": "convert_jpg", "name": "Convert JPG to PNG", "icon": "image"},
    {"id": "prepare", "name": "Prepare & Resize", "icon": "image"},
    {"id": "annotate", "name": "Annotate (MiniMax M3)", "icon": "brain"},
    {"id": "quality_review", "name": "Quality Review", "icon": "check"},
    {"id": "validate", "name": "Validate LaTeX", "icon": "check"},
    {"id": "build", "name": "Build JSONL Dataset", "icon": "database"},
    {"id": "split", "name": "Train/Val/Test Split", "icon": "split"},
]

state_lock = threading.Lock()

pipeline_state = {
    "running": False,
    "paused": False,
    "stop_requested": False,
    "current_stage": None,
    "completed_stages": [],
    "stages": {},
    "started_at": None,
    "finished_at": None,
    "config": {},
}

log_buffer = deque(maxlen=5000)
log_event_id = 0
log_lock = threading.Lock()


def dirs():
    d = {
        "crop_pii": WORK_DIR / "1_cropped",
        "filter_blank": WORK_DIR / "2_filtered",
        "filter_blank_rejected": WORK_DIR / "2_filtered_rejected",
        "deskew": WORK_DIR / "3_deskewed",
        "convert_jpg": WORK_DIR / "4_converted_png",
        "prepare": WORK_DIR / "5_prepared",
        "annotate": WORK_DIR / "6_annotations",
        "quality_review": WORK_DIR / "6_quality_reviewed",
        "quality_flagged": WORK_DIR / "6_flagged_for_review",
        "validate": WORK_DIR / "7_validated",
        "validate_rejected": WORK_DIR / "7_rejected",
        "build": WORK_DIR / "8_dataset",
        "split": WORK_DIR / "9_split",
    }
    return d


def init_stage(stage_id, total, baseline=0):
    with state_lock:
        pipeline_state["stages"][stage_id] = {
            "status": "running",
            "total": total,
            "baseline": baseline,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "started_at": time.time(),
            "finished_at": None,
            "eta_seconds": None,
            "pct": 0.0,
            "items_per_sec": 0.0,
            "extra": {},
        }
        pipeline_state["current_stage"] = stage_id


def update_stage(stage_id, processed=None, success=None, failed=None, skipped=None, extra=None):
    with state_lock:
        s = pipeline_state["stages"].get(stage_id)
        if not s:
            return
        if processed is not None:
            s["processed"] = processed
        if success is not None:
            s["success"] = success
        if failed is not None:
            s["failed"] = failed
        if skipped is not None:
            s["skipped"] = skipped
        if extra:
            s["extra"].update(extra)
        if s["total"] > 0:
            s["pct"] = round(s["processed"] / s["total"] * 100, 1)
        elapsed = time.time() - s["started_at"]
        done_this_run = s["processed"] - s.get("baseline", 0)
        if done_this_run > 0 and elapsed > 0:
            rate = done_this_run / elapsed
            s["items_per_sec"] = round(rate, 2)
            remaining = s["total"] - s["processed"]
            s["eta_seconds"] = round(remaining / rate) if rate > 0 else None


def finish_stage(stage_id, status="completed"):
    with state_lock:
        s = pipeline_state["stages"].get(stage_id)
        if s:
            s["status"] = status
            s["finished_at"] = time.time()
            s["pct"] = 100.0 if status == "completed" else s["pct"]
        if stage_id not in pipeline_state["completed_stages"]:
            pipeline_state["completed_stages"].append(stage_id)


def add_log(msg, level="info"):
    global log_event_id
    with log_lock:
        log_event_id += 1
        entry = {
            "id": log_event_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "msg": msg,
            "level": level,
            "stage": pipeline_state.get("current_stage", ""),
        }
        log_buffer.append(entry)


def get_image_files(directory, extensions=None):
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    d = Path(directory)
    if not d.exists():
        return []
    files = sorted([f for f in d.iterdir() if f.suffix.lower() in extensions])
    return files


def get_paper_bg_color(img, margin=40):
    h, w = img.shape[:2]
    margin = min(margin, h // 4, w // 4)
    r1 = img[-margin:, :margin]
    r2 = img[-margin:, -margin:]
    gray_avg = (cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY).mean() + cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY).mean()) / 2.0
    bg_val = int(round(gray_avg))
    return (bg_val, bg_val, bg_val)


def get_skew_angle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    edges = cv2.Canny(eroded, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=int(img.shape[1] * 0.15), maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        if -8 <= angle <= 8:
            angles.append(angle)
    if not angles:
        return 0.0
    skew = float(np.median(angles))
    if abs(skew) > 5.0:
        return 0.0
    return skew


def rotate_image(img, angle, bg_color=(255, 255, 255)):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos_v, sin_v = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int((h * sin_v) + (w * cos_v))
    new_h = int((h * cos_v) + (w * sin_v))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=bg_color)


def crop_single(args):
    src, out, pct = args
    try:
        img = cv2.imread(src)
        if img is None:
            return ("fail", src, "cannot read image")
        h, w = img.shape[:2]
        crop_h = int(h * pct / 100)
        bg_color = get_paper_bg_color(img)
        cv2.rectangle(img, (0, 0), (w, crop_h), bg_color, -1)
        if out.lower().endswith(".png"):
            cv2.imwrite(out, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return ("ok", src, "")
    except Exception as e:
        return ("fail", src, str(e))


def filter_single(args):
    src, keep_dir, rej_dir, dark_thresh, max_dark_pct = args
    try:
        img = cv2.imread(src)
        if img is None:
            return ("fail", src, "cannot read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dark_pct = (np.sum(gray < dark_thresh) / (h * w)) * 100
        name = Path(src).name
        if dark_pct <= max_dark_pct:
            shutil.copy2(src, str(Path(rej_dir) / name))
            return ("rejected", src, "")
        shutil.copy2(src, str(Path(keep_dir) / name))
        return ("kept", src, "")
    except Exception as e:
        return ("fail", src, str(e))


def deskew_single(args):
    src, out = args
    try:
        img = cv2.imread(src)
        if img is None:
            return ("fail", src, "cannot read image")
        bg_color = (int(np.mean(img[:, :, 2])), int(np.mean(img[:, :, 1])), int(np.mean(img[:, :, 0])))
        angle = get_skew_angle(img)
        rotated = False
        if abs(angle) >= 0.8:
            img = rotate_image(img, angle, bg_color)
            rotated = True
        cv2.imwrite(out, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        return ("rotated" if rotated else "ok", src, "")
    except Exception as e:
        return ("fail", src, str(e))


def convert_single(args):
    src, out = args
    try:
        img = Image.open(src)
        if img.mode == "CMYK":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "RGBA", "LA"):
            img = img.convert("RGB")
        img.save(out, "PNG", optimize=True)
        return ("ok", src, "")
    except Exception as e:
        return ("fail", src, str(e))


def prepare_single(args):
    src, out, max_size = args
    try:
        img = Image.open(src)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max_size > 0:
            w, h = img.size
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        img.save(out, "PNG")
        return ("ok", src, "", list(img.size))
    except Exception as e:
        return ("fail", src, str(e), None)


def default_cpu_workers():
    return max(2, (os.cpu_count() or 4) - 1)


ANNOTATION_SYSTEM_PROMPT = """You are an expert OCR system specialized in reading handwritten mathematics from undergraduate-level answer sheets.

Your task: Convert ONLY the handwritten mathematical content into a complete, compilable LaTeX document.

INCLUDE:
- All handwritten equations and mathematical expressions
- All handwritten text that is part of the solution (like "Solution:", "Let x =", etc.)
- Preserve the spatial layout and logical flow exactly as written

IGNORE (do NOT include these in output):
- Any printed/typed text (headers, footers, instructions)
- Student name, roll number, date, page numbers
- Cancelled/crossed-out work
- Rough/scratch work sections
- Any watermarks or stamps

FORMATTING RULES:
1. Output must be a COMPLETE LaTeX document (\\documentclass through \\end{document})
2. Use amsmath, amssymb, amsfonts packages
3. Preserve the exact spatial layout
4. Use appropriate environments: align for multi-line equations, equation for single display equations
5. Inline math uses $...$, display math uses \\[...\\] or align/equation environments
6. If something is illegible, mark as \\textit{[illegible]}
7. The output MUST compile without errors using pdflatex

Output ONLY the LaTeX code. No explanations, no markdown, no comments."""

ANNOTATION_USER_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into LaTeX. Ignore all printed text, student info, page numbers, "
    "cancelled work, and rough work. Preserve the spatial layout exactly. "
    "Output a complete compilable LaTeX document only."
)

TRAINING_SYSTEM_PROMPT = """You are an expert OCR system specialized in reading handwritten mathematics from undergraduate-level answer sheets.

Your task: Convert ONLY the handwritten mathematical content into a complete, compilable LaTeX document.

INCLUDE:
- All handwritten equations and mathematical expressions
- All handwritten text that is part of the solution (like "Solution:", "Let x =", etc.)
- Preserve the spatial layout and logical flow exactly as written

IGNORE (do NOT include these in output):
- Any printed/typed text (headers, footers, instructions)
- Student name, roll number, date, page numbers
- Cancelled/crossed-out work
- Rough/scratch work sections
- Any watermarks or stamps

FORMATTING RULES:
1. Output must be a COMPLETE LaTeX document (\\documentclass through \\end{document})
2. Use amsmath, amssymb, amsfonts packages
3. Preserve the exact spatial layout
4. Use appropriate environments: align for multi-line equations, equation for single display equations
5. Inline math uses $...$, display math uses \\[...\\] or align/equation environments
6. If something is illegible, mark as \\textit{[illegible]}
7. The output MUST compile without errors using pdflatex

Output ONLY the LaTeX code. No explanations, no markdown, no comments."""

TRAINING_USER_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into a complete, compilable LaTeX document. Ignore printed text, "
    "student info, page numbers, cancelled work and rough work. Output only LaTeX."
)


def encode_image_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def strip_reasoning(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"<think>", text, re.IGNORECASE):
        # Unclosed think tag (response truncated mid-reasoning): keep whatever
        # follows the tag; validity check rejects it if no document survived
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


def is_valid_latex(latex):
    bad = ["please provide the image", "please attach", "no handwritten content provided", "i cannot see", "no image", "unable to process"]
    lower = latex.lower()
    for phrase in bad:
        if phrase in lower:
            return False
    if "\\documentclass" not in latex or "\\end{document}" not in latex:
        return False
    if "<think>" in lower:
        return False
    return True


REQUIRED_PACKAGES = [r"\usepackage{amsmath}", r"\usepackage{amssymb}", r"\usepackage{amsfonts}"]
COMMAND_PACKAGE_MAP = {
    "cancel": [r"\cancel", r"\bcancel", r"\xcancel", r"\cancelto"],
    "mathtools": [r"\xmapsto", r"\coloneqq", r"\xLeftrightarrow"],
    "gensymb": [r"\degree", r"\celsius"],
    "textcomp": [r"\textcelsius"],
    "graphicx": [r"\includegraphics"],
    "enumitem": [r"\begin{enumerate}["],
    "ulem": [r"\sout", r"\uline"],
}


def looks_truncated(latex):
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
    # count only display-math \[ \] — a lookbehind excludes the \[ inside
    # line-break spacing like \\[2mm], which is not a math delimiter
    if len(re.findall(r"(?<!\\)\\\[", body)) != len(re.findall(r"(?<!\\)\\\]", body)):
        return True
    begins = len(re.findall(r"\\begin\{", body))
    ends = len(re.findall(r"\\end\{", body))
    if begins != ends:
        return True
    return False


def auto_fix_latex(latex):
    fixed = latex.strip()
    if fixed.startswith("```"):
        fixed = re.sub(r"^```[a-zA-Z]*\n?", "", fixed)
        fixed = re.sub(r"\n?```$", "", fixed).strip()
    fixed = fixed.replace("% REVIEWED: OK", "").strip()
    fixed = re.sub(r"\\require\{[^}]*\}", "", fixed)
    fixed = re.sub(r"\\textcircled\{([^}]*)\}", r"(\1)", fixed)
    for i in range(20):
        fixed = fixed.replace(chr(0x2460 + i), f"({i + 1})")
    fixed = fixed.replace("\u2212", "-")
    if "\\documentclass" not in fixed:
        fixed = "\\documentclass{article}\n" + fixed
    needed_pkgs = list(REQUIRED_PACKAGES)
    for pkg_name, triggers in COMMAND_PACKAGE_MAP.items():
        if any(t in fixed for t in triggers):
            needed_pkgs.append(r"\usepackage{" + pkg_name + "}")
    lines = fixed.splitlines()
    preamble_text = fixed.split("\\begin{document}")[0] if "\\begin{document}" in fixed else fixed
    missing_pkgs = []
    for pkg in needed_pkgs:
        pkg_name_m = re.search(r"\{(\w+)\}", pkg)
        if pkg_name_m:
            pn = pkg_name_m.group(1)
            if f"{{{pn}}}" not in preamble_text:
                missing_pkgs.append(pkg)
    if missing_pkgs:
        insert_idx = 0
        for i, line in enumerate(lines):
            if "\\documentclass" in line:
                insert_idx = i + 1
                break
        for pkg in reversed(missing_pkgs):
            lines.insert(insert_idx, pkg)
        fixed = "\n".join(lines)
    if "\\begin{document}" not in fixed:
        usepackage_positions = [m.end() for m in re.finditer(r"\\usepackage(\[[^\]]*\])?\{[^}]*\}", fixed)]
        if usepackage_positions:
            pos = usepackage_positions[-1]
            fixed = fixed[:pos] + "\n\n\\begin{document}\n" + fixed[pos:]
        else:
            fixed += "\n\\begin{document}\n"
    if "\\end{document}" not in fixed:
        fixed = fixed.rstrip() + "\n\n\\end{document}\n"
    return fixed


def compile_latex_check(latex, timeout=30):
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "doc.tex"
        tex_file.write_text(latex, encoding="utf-8")
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "doc.tex"],
                cwd=tmpdir, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except FileNotFoundError:
            return False, "PDFLATEX_NOT_FOUND"
        pdf_file = Path(tmpdir) / "doc.pdf"
        if result.returncode == 0 and pdf_file.exists():
            return True, ""
        log_file = Path(tmpdir) / "doc.log"
        err = "Unknown error"
        if log_file.exists():
            log_text = log_file.read_text(encoding="utf-8", errors="ignore")
            for line in log_text.splitlines():
                if line.startswith("!"):
                    err = line.strip()
                    break
        return False, err


def validate_single_tex(tex_path_str):
    tex_path = Path(tex_path_str)
    original = tex_path.read_text(encoding="utf-8")
    result = {"file": tex_path.name, "status": None, "error": "", "rewritten": False}
    if looks_truncated(original):
        result["status"] = "TRUNCATED"
        result["error"] = "Output appears cut off / unbalanced delimiters"
        return result
    ok, err = compile_latex_check(original)
    if ok:
        result["status"] = "PASS"
        return result
    if err == "PDFLATEX_NOT_FOUND":
        result["status"] = "SKIP"
        result["error"] = "pdflatex not found - skipping compile validation"
        return result
    fixed = auto_fix_latex(original)
    if fixed != original:
        ok2, err2 = compile_latex_check(fixed)
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


def stage_crop_pii(config):
    stage_id = "crop_pii"
    d = dirs()
    d["crop_pii"].mkdir(parents=True, exist_ok=True)
    source = Path(config.get("source_dir", str(DATA_DIR)))
    
    if not source.exists():
        add_log(f"ERROR: Source directory not found: {source}", "error")
        finish_stage(stage_id, "failed")
        return False
    
    if source.resolve() == WORK_DIR.resolve() or WORK_DIR.resolve() in [p.resolve() for p in source.parents]:
        add_log(f"ERROR: Source cannot be inside workspace directory", "error")
        finish_stage(stage_id, "failed")
        return False
    
    files = get_image_files(source)
    total = len(files)
    pct = config.get("crop_pct", 10)
    workers = int(config.get("cpu_workers", default_cpu_workers()))
    pending = [f for f in files if not (d["crop_pii"] / f.name).exists()]
    already = total - len(pending)
    add_log(f"Crop PII: {total} images, {already} done, {len(pending)} pending, {workers} workers")
    add_log(f"  -> Reading from: {source}")
    add_log(f"  -> Writing to: {d['crop_pii']}")
    init_stage(stage_id, total, baseline=already)
    success = already
    failed = 0
    processed = already
    update_stage(stage_id, processed=processed, success=success, skipped=already)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(crop_single, (str(f), str(d["crop_pii"] / f.name), pct)) for f in pending]
        for fut in as_completed(futures):
            if pipeline_state["stop_requested"]:
                add_log("Stop requested. Halting.", "warn")
                ex.shutdown(wait=True, cancel_futures=True)
                finish_stage(stage_id, "stopped")
                return False
            status, src, err = fut.result()
            if status == "ok":
                success += 1
            else:
                failed += 1
                add_log(f"Crop PII FAIL: {Path(src).name} - {err}", "error")
            processed += 1
            update_stage(stage_id, processed=processed, success=success, failed=failed)
            if processed % 500 == 0:
                add_log(f"Crop PII: [{processed}/{total}]")
    finish_stage(stage_id)
    add_log(f"Crop PII complete: {success} ok, {failed} failed out of {total}")
    add_log(f"  Original data UNTOUCHED in: {source}")
    return True


def stage_deskew(config):
    stage_id = "deskew"
    d = dirs()
    d["deskew"].mkdir(parents=True, exist_ok=True)
    source = d["filter_blank"]
    files = get_image_files(source)
    total = len(files)
    workers = int(config.get("cpu_workers", default_cpu_workers()))
    pending = [f for f in files if not (d["deskew"] / f.name).exists()]
    already = total - len(pending)
    add_log(f"Deskew: {total} images, {already} done, {len(pending)} pending, {workers} workers")
    init_stage(stage_id, total, baseline=already)
    success = already
    failed = 0
    rotated_count = 0
    processed = already
    update_stage(stage_id, processed=processed, success=success, skipped=already)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(deskew_single, (str(f), str(d["deskew"] / f.name))) for f in pending]
        for fut in as_completed(futures):
            if pipeline_state["stop_requested"]:
                ex.shutdown(wait=True, cancel_futures=True)
                finish_stage(stage_id, "stopped")
                return False
            status, src, err = fut.result()
            if status in ("ok", "rotated"):
                success += 1
                if status == "rotated":
                    rotated_count += 1
            else:
                failed += 1
                add_log(f"Deskew FAIL: {Path(src).name} - {err}", "error")
            processed += 1
            update_stage(stage_id, processed=processed, success=success, failed=failed, extra={"rotated": rotated_count})
            if processed % 500 == 0:
                add_log(f"Deskew: [{processed}/{total}] rotated={rotated_count}")
    finish_stage(stage_id)
    add_log(f"Deskew complete: {success} ok, {rotated_count} rotated, {failed} failed")
    return True


def stage_filter_blank(config):
    stage_id = "filter_blank"
    d = dirs()
    d["filter_blank"].mkdir(parents=True, exist_ok=True)
    d["filter_blank_rejected"].mkdir(parents=True, exist_ok=True)
    source = d["crop_pii"]
    files = get_image_files(source)
    total = len(files)
    dark_thresh = config.get("dark_thresh", 210)
    max_dark_pct = config.get("max_dark_pct", 0.5)
    workers = int(config.get("cpu_workers", default_cpu_workers()))
    pending = [f for f in files if not (d["filter_blank"] / f.name).exists() and not (d["filter_blank_rejected"] / f.name).exists()]
    already = total - len(pending)
    add_log(f"Filter: {total} images, {already} done, {len(pending)} pending, {workers} workers")
    init_stage(stage_id, total, baseline=already)
    kept = 0
    rejected = 0
    failed = 0
    processed = already
    update_stage(stage_id, processed=processed, skipped=already)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(filter_single, (str(f), str(d["filter_blank"]), str(d["filter_blank_rejected"]), dark_thresh, max_dark_pct))
            for f in pending
        ]
        for fut in as_completed(futures):
            if pipeline_state["stop_requested"]:
                ex.shutdown(wait=True, cancel_futures=True)
                finish_stage(stage_id, "stopped")
                return False
            status, src, err = fut.result()
            if status == "kept":
                kept += 1
            elif status == "rejected":
                rejected += 1
            else:
                failed += 1
                add_log(f"Filter FAIL: {Path(src).name} - {err}", "error")
            processed += 1
            update_stage(stage_id, processed=processed, success=kept, failed=failed, extra={"kept": kept, "rejected": rejected})
            if processed % 500 == 0:
                add_log(f"Filter: [{processed}/{total}] kept={kept} rejected={rejected}")
    finish_stage(stage_id)
    add_log(f"Filter complete: {kept} kept, {rejected} blank rejected, {failed} errors")
    return True


def stage_convert_jpg(config):
    stage_id = "convert_jpg"
    d = dirs()
    d["convert_jpg"].mkdir(parents=True, exist_ok=True)
    source = d["deskew"]
    files = get_image_files(source)
    total = len(files)
    workers = int(config.get("cpu_workers", default_cpu_workers()))
    pending = [f for f in files if not (d["convert_jpg"] / (f.stem + ".png")).exists()]
    already = total - len(pending)
    add_log(f"Convert JPG->PNG: {total} images, {already} done, {len(pending)} pending, {workers} workers")
    init_stage(stage_id, total, baseline=already)
    success = already
    failed = 0
    processed = already
    update_stage(stage_id, processed=processed, success=success, skipped=already)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(convert_single, (str(f), str(d["convert_jpg"] / (f.stem + ".png")))) for f in pending]
        for fut in as_completed(futures):
            if pipeline_state["stop_requested"]:
                ex.shutdown(wait=True, cancel_futures=True)
                finish_stage(stage_id, "stopped")
                return False
            status, src, err = fut.result()
            if status == "ok":
                success += 1
            else:
                failed += 1
                add_log(f"Convert FAIL: {Path(src).name} - {err}", "error")
            processed += 1
            update_stage(stage_id, processed=processed, success=success, failed=failed)
            if processed % 500 == 0:
                add_log(f"Convert: [{processed}/{total}]")
    finish_stage(stage_id)
    add_log(f"Convert complete: {success} ok, {failed} failed out of {total}")
    return True


def stage_prepare(config):
    stage_id = "prepare"
    d = dirs()
    d["prepare"].mkdir(parents=True, exist_ok=True)
    source = d["convert_jpg"]
    files = sorted(source.glob("*.png"))
    total = len(files)
    max_size = config.get("max_size", 2048)
    workers = int(config.get("cpu_workers", default_cpu_workers()))
    manifest_path = WORK_DIR / "image_manifest.json"
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            add_log(f"Prepare: could not read manifest ({e}); starting fresh", "warn")
            manifest = []
    known = {m["original"] for m in manifest}
    next_idx = 0
    for m in manifest:
        try:
            next_idx = max(next_idx, int(m["id"].split("_")[1]))
        except Exception:
            pass
    # ids are assigned from the manifest, never from sort position, so adding
    # new source batches cannot shift existing image<->annotation pairings
    new_files = [f for f in files if f.name not in known]
    already = total - len(new_files)
    add_log(f"Prepare: {total} converted, {already} already in manifest, {len(new_files)} new, max_size={max_size}, {workers} workers")
    init_stage(stage_id, total, baseline=already)
    success = already
    failed = 0
    processed = already
    update_stage(stage_id, processed=processed, success=success, skipped=already)

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tasks = []
    for f in new_files:
        next_idx += 1
        page_id = f"page_{next_idx:05d}"
        tasks.append((str(f), str(d["prepare"] / f"{page_id}.png"), max_size, page_id, f.name))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(prepare_single, (t[0], t[1], t[2])): t for t in tasks}
        for fut in as_completed(futures):
            if pipeline_state["stop_requested"]:
                ex.shutdown(wait=True, cancel_futures=True)
                save_manifest()
                finish_stage(stage_id, "stopped")
                return False
            _, _, _, page_id, orig_name = futures[fut]
            status, src, err, size = fut.result()
            if status == "ok":
                manifest.append({"id": page_id, "original": orig_name, "size": size})
                success += 1
            else:
                failed += 1
                add_log(f"Prepare FAIL: {orig_name} - {err}", "error")
            processed += 1
            update_stage(stage_id, processed=processed, success=success, failed=failed)
            if processed % 500 == 0:
                add_log(f"Prepare: [{processed}/{total}]")
                save_manifest()
    save_manifest()
    finish_stage(stage_id)
    add_log(f"Prepare complete: {success} ok, {failed} failed. Manifest: {len(manifest)} entries.")
    return True


def stage_annotate(config):
    stage_id = "annotate"
    d = dirs()
    d["annotate"].mkdir(parents=True, exist_ok=True)
    source = d["prepare"]
    files = sorted(source.glob("*.png"))
    pending = [f for f in files if not (d["annotate"] / f"{f.stem}.tex").exists()]
    already_done = len(files) - len(pending)
    total = len(files)
    workers = max(1, int(config.get("annotate_workers", 8)))
    add_log(f"Annotate: {total} total, {already_done} done, {len(pending)} pending, {workers} workers")
    init_stage(stage_id, total, baseline=already_done)
    api_base = config.get("api_base_url", "https://api.tokenrouter.com/v1")
    model = config.get("model", "MiniMax-M3")
    delay = config.get("annotate_delay", 0)
    max_retries = config.get("max_retries", 3)
    max_tokens = int(config.get("max_tokens", 8192))
    temperature = float(config.get("temperature", 0.1))
    keys_cfg = config.get("api_keys") or config.get("api_key", "")
    if isinstance(keys_cfg, str):
        api_keys = [k.strip() for k in keys_cfg.split(",") if k.strip()]
    else:
        api_keys = [k.strip() for k in keys_cfg if k and k.strip()]
    key_limit = int(config.get("api_key_limit", 0) or 0)
    if key_limit > 0:
        api_keys = api_keys[:key_limit]
    if not api_keys:
        add_log("No API key configured", "error")
        finish_stage(stage_id, "failed")
        return False
    try:
        from openai import OpenAI
        clients = [OpenAI(base_url=api_base, api_key=k) for k in api_keys]
    except Exception as e:
        add_log(f"Failed to create API client: {e}", "error")
        finish_stage(stage_id, "failed")
        return False
    add_log(f"Annotate: {len(clients)} API key(s), max_tokens={max_tokens}")

    invalid_raw_dir = WORK_DIR / "6_invalid_raw"
    invalid_raw_dir.mkdir(parents=True, exist_ok=True)
    quarantine = bool(config.get("quarantine_new", False))
    if quarantine:
        add_log("Quarantine mode: pages annotated in this run will be EXCLUDED from the main dataset", "warn")
    new_stems = []
    counters = {"success": already_done, "failed": 0, "skipped": 0, "processed": already_done}
    counters_lock = threading.Lock()
    pause_until = [0.0]
    backoff_step = [0]
    pause_lock = threading.Lock()
    fatal = {"reason": None}

    def wait_if_paused():
        while True:
            if pipeline_state["stop_requested"] or fatal["reason"]:
                return False
            with pause_lock:
                wait_s = pause_until[0] - time.time()
            if wait_s <= 0:
                return True
            time.sleep(min(wait_s, 1.0))

    def trigger_backoff():
        with pause_lock:
            if time.time() >= pause_until[0]:
                backoff_step[0] = min(backoff_step[0] + 1, 4)
                wait = min(60 * (2 ** (backoff_step[0] - 1)), 300)
                pause_until[0] = time.time() + wait
                add_log(f"Rate limited! All workers backing off {wait}s...", "warn")

    def annotate_one(task_idx, img_path):
        client = clients[task_idx % len(clients)]
        retries = 0
        while retries < max_retries:
            if not wait_if_paused():
                return "aborted"
            try:
                img_b64 = encode_image_b64(str(img_path))
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                                {"type": "text", "text": ANNOTATION_USER_PROMPT},
                            ],
                        },
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                msg = response.choices[0].message
                raw = getattr(msg, "content", None) or ""
                latex = clean_latex_response(raw)
                if is_valid_latex(latex):
                    tex_file = d["annotate"] / f"{img_path.stem}.tex"
                    tex_file.write_text(latex, encoding="utf-8")
                    if quarantine:
                        with counters_lock:
                            new_stems.append(img_path.stem)
                    with pause_lock:
                        backoff_step[0] = 0
                    if delay > 0:
                        time.sleep(delay)
                    return "success"
                try:
                    (invalid_raw_dir / f"{img_path.stem}.txt").write_text(raw, encoding="utf-8")
                except Exception:
                    pass
                add_log(f"Annotate SKIP (invalid output): {img_path.name} (raw saved)", "warn")
                return "skipped"
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    trigger_backoff()
                elif "402" in err_str or "insufficient" in err_str.lower() or "credit" in err_str.lower():
                    fatal["reason"] = "CREDIT EXHAUSTED"
                    return "fatal"
                elif "401" in err_str or "unauthorized" in err_str.lower():
                    fatal["reason"] = "INVALID API KEY"
                    return "fatal"
                else:
                    retries += 1
                    if retries < max_retries:
                        add_log(f"Retry {retries}/{max_retries}: {img_path.name} - {err_str[:80]}", "warn")
                        time.sleep(2)
                    else:
                        add_log(f"Annotate FAIL: {img_path.name} - {err_str[:100]}", "error")
                        return "failed"
        return "failed"

    aborted = False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(annotate_one, i, p): p for i, p in enumerate(pending)}
        for fut in as_completed(futures):
            img_path = futures[fut]
            try:
                outcome = fut.result()
            except Exception as e:
                outcome = "failed"
                add_log(f"Annotate worker crashed: {img_path.name} - {e}", "error")
            with counters_lock:
                if outcome in ("success", "skipped", "failed"):
                    counters[outcome] += 1
                    counters["processed"] += 1
                snap = dict(counters)
            update_stage(stage_id, processed=snap["processed"], success=snap["success"], failed=snap["failed"], skipped=snap["skipped"])
            if outcome == "success" and snap["processed"] % 10 == 0:
                add_log(f"Annotate: [{snap['processed']}/{total}] {img_path.name}")
            if pipeline_state["stop_requested"] or fatal["reason"]:
                aborted = True
                ex.shutdown(wait=True, cancel_futures=True)
                break
    if quarantine and new_stems:
        excl_path = WORK_DIR / "excluded_pages.json"
        try:
            data = json.loads(excl_path.read_text(encoding="utf-8")) if excl_path.exists() else {"pages": []}
        except Exception:
            data = {"pages": []}
        merged = sorted(set(data.get("pages", [])) | set(new_stems))
        data["pages"] = merged
        data["reason"] = data.get("reason", "quarantined re-annotations")
        excl_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        add_log(f"Quarantine: {len(new_stems)} new pages added to excluded_pages.json ({len(merged)} total excluded)")
    if fatal["reason"]:
        add_log(f"{fatal['reason']}! Stopping annotation.", "error")
        finish_stage(stage_id, "failed")
        return False
    if pipeline_state["stop_requested"] or aborted:
        finish_stage(stage_id, "stopped")
        return False
    finish_stage(stage_id)
    snap = dict(counters)
    add_log(f"Annotate complete: {snap['success']} ok, {snap['failed']} failed, {snap['skipped']} skipped out of {total}")
    return True


def stage_quality_review(config):
    stage_id = "quality_review"
    d = dirs()
    ann_dir = d["annotate"]
    flagged_dir = d["quality_flagged"]
    out_dir = d["quality_review"]
    # derived dirs: clear on each run so deleted/re-annotated sources can't
    # leave stale copies behind
    for derived in (flagged_dir, out_dir):
        if derived.exists():
            shutil.rmtree(derived)
        derived.mkdir(parents=True, exist_ok=True)
    tex_files = sorted(ann_dir.glob("*.tex"))
    total = len(tex_files)
    add_log(f"Quality Review: Analyzing {total} annotations")
    init_stage(stage_id, total)
    
    quality_issues = []
    duplicates = {}
    lengths = []
    done = 0
    
    for tex_file in tex_files:
        if pipeline_state["stop_requested"]:
            finish_stage(stage_id, "stopped")
            return False
        
        content = tex_file.read_text(encoding="utf-8")
        stem = tex_file.stem
        issues = []
        
        lengths.append(len(content))
        
        if len(content) < 300:
            issues.append("too_short")
        
        if len(content) > 15000:
            issues.append("too_long")
        
        body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", content, re.DOTALL)
        if body_match:
            body = body_match.group(1)
            math_content = re.findall(r"\$[^$]+\$|\\\[[^\\]+\\\]|\\begin\{[^}]+\}.*?\\end\{[^}]+\}", body, re.DOTALL)
            if not math_content:
                issues.append("no_math_detected")
        
        dollar_count = len(re.findall(r"(?<!\\)\$", content))
        if dollar_count % 2 != 0:
            issues.append("unbalanced_dollars")
        
        begin_count = len(re.findall(r"\\begin\{", content))
        end_count = len(re.findall(r"\\end\{", content))
        if begin_count != end_count:
            issues.append("unbalanced_environments")
        
        hallucination_patterns = [
            r"\\text\{.*?illegible.*?\}",
            r"\\textit\{.*?illegible.*?\}",
            r"\[.*?illegible.*?\]",
        ]
        for pattern in hallucination_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append("contains_illegible_marker")
                break
        
        if "<think>" in content.lower():
            issues.append("contains_think_block")

        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in duplicates:
            issues.append(f"duplicate_of_{duplicates[content_hash]}")
        else:
            duplicates[content_hash] = stem

        if issues:
            quality_issues.append({"file": stem, "issues": issues})
            shutil.copy2(str(tex_file), str(flagged_dir / tex_file.name))
        # illegible markers are prompt-sanctioned, so they're flagged for human
        # review but still flow to validation; everything else is excluded
        blocking = [i for i in issues if i != "contains_illegible_marker"]
        if not blocking:
            shutil.copy2(str(tex_file), str(out_dir / tex_file.name))

        done += 1
        update_stage(stage_id, processed=done, success=done - len(quality_issues), failed=len(quality_issues))
        
        if done % 50 == 0 or done == total:
            add_log(f"Quality Review: [{done}/{total}] issues={len(quality_issues)}")
    
    if lengths:
        avg_len = sum(lengths) // len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        add_log(f"  Length stats: min={min_len}, max={max_len}, avg={avg_len}")
    
    add_log(f"  Quality issues found: {len(quality_issues)}")
    add_log(f"  Flagged for review: {flagged_dir}")
    
    report = {
        "total": total,
        "issues_count": len(quality_issues),
        "issues": quality_issues,
        "length_stats": {"min": min(lengths) if lengths else 0, "max": max(lengths) if lengths else 0, "avg": avg_len if lengths else 0},
    }
    report_path = WORK_DIR / "quality_review_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    add_log(f"  Report saved: {report_path}")
    
    finish_stage(stage_id)
    return True


def stage_validate(config):
    stage_id = "validate"
    d = dirs()
    src_dir = d["quality_review"]
    out_dir = d["validate"]
    rej_dir = d["validate_rejected"]
    for derived in (out_dir, rej_dir):
        if derived.exists():
            shutil.rmtree(derived)
        derived.mkdir(parents=True, exist_ok=True)
    tex_files = sorted(src_dir.glob("*.tex"))
    total = len(tex_files)
    add_log(f"Validate: Found {total} .tex files in {src_dir}")
    init_stage(stage_id, total)
    workers = config.get("validate_workers", 4)
    counts = {"PASS": 0, "FIXED": 0, "FAIL": 0, "TRUNCATED": 0, "SKIP": 0}
    done = 0
    has_pdflatex = True
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(validate_single_tex, str(f)): f for f in tex_files}
            for fut in as_completed(futures):
                if pipeline_state["stop_requested"]:
                    finish_stage(stage_id, "stopped")
                    return False
                res = fut.result()
                status = res["status"]
                if status == "SKIP":
                    counts["SKIP"] = counts.get("SKIP", 0) + 1
                    has_pdflatex = False
                else:
                    counts[status] = counts.get(status, 0) + 1
                src_file = src_dir / res["file"]
                if status in ("PASS", "FIXED"):
                    if status == "FIXED" and res.get("rewritten"):
                        shutil.copy2(str(src_file), str(out_dir / res["file"]))
                    else:
                        shutil.copy2(str(src_file), str(out_dir / res["file"]))
                elif status in ("FAIL", "TRUNCATED"):
                    if src_file.exists():
                        shutil.copy2(str(src_file), str(rej_dir / res["file"]))
                done += 1
                update_stage(stage_id, processed=done, success=counts["PASS"] + counts["FIXED"], failed=counts["FAIL"] + counts["TRUNCATED"], skipped=counts.get("SKIP", 0), extra=counts)
                if done % 25 == 0 or done == total:
                    add_log(f"Validate: [{done}/{total}] PASS={counts['PASS']} FIXED={counts['FIXED']} FAIL={counts['FAIL']} TRUNC={counts['TRUNCATED']}")
    except Exception as e:
        add_log(f"Validate error: {e}", "error")
    finish_stage(stage_id)
    kept = counts["PASS"] + counts["FIXED"]
    add_log(f"Validate complete: {kept} kept -> {out_dir}, {counts['FAIL']+counts['TRUNCATED']} rejected -> {rej_dir}")
    if not has_pdflatex:
        add_log("pdflatex not found - compile validation skipped, all files kept", "warn")
    return True


def stage_build(config):
    stage_id = "build"
    d = dirs()
    d["build"].mkdir(parents=True, exist_ok=True)
    images_dir = d["prepare"]
    annotations_dir = d["validate"]
    output_file = d["build"] / "full_dataset.jsonl"
    add_log(f"Build: Combining images + annotations")
    init_stage(stage_id, 1)
    annotation_files = sorted(annotations_dir.glob("*.tex"))
    excluded = set()
    excl_path = WORK_DIR / "excluded_pages.json"
    if excl_path.exists():
        try:
            excluded = set(json.loads(excl_path.read_text(encoding="utf-8")).get("pages", []))
            add_log(f"Build: excluding {len(excluded)} pages (excluded_pages.json)")
        except Exception as e:
            add_log(f"Build: could not read excluded_pages.json: {e}", "warn")
    samples = []
    skipped = 0
    for ann_file in annotation_files:
        if ann_file.stem in excluded:
            skipped += 1
            continue
        image_name = f"{ann_file.stem}.png"
        image_file = images_dir / image_name
        if not image_file.exists():
            skipped += 1
            continue
        latex_content = ann_file.read_text(encoding="utf-8").strip()
        if not latex_content or len(latex_content) < 200:
            skipped += 1
            continue
        refusal_phrases = ["please provide the image", "no handwritten content", "please attach", "i cannot see"]
        if any(p in latex_content.lower() for p in refusal_phrases):
            skipped += 1
            continue
        latex_content = latex_content.replace("% REVIEWED: OK", "").strip()
        sample = {
            "id": ann_file.stem,
            "image": f"images/{image_name}",
            "conversations": [
                {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                {"role": "user", "content": TRAINING_USER_PROMPT},
                {"role": "assistant", "content": latex_content},
            ],
        }
        samples.append(sample)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    update_stage(stage_id, processed=1, success=1, extra={"samples": len(samples), "skipped": skipped})
    if samples:
        lengths = [len(s["conversations"][2]["content"]) for s in samples]
        add_log(f"Build: {len(samples)} samples, avg LaTeX {sum(lengths)//len(lengths)} chars, skipped {skipped}")
    finish_stage(stage_id)
    add_log(f"Build complete: {len(samples)} samples -> {output_file}")
    return True


def stage_split(config):
    stage_id = "split"
    d = dirs()
    input_file = d["build"] / "full_dataset.jsonl"
    output_dir = d["split"]
    add_log(f"Split: Reading {input_file}")
    init_stage(stage_id, 1)
    if not input_file.exists():
        add_log(f"Split: Input file not found: {input_file}", "error")
        finish_stage(stage_id, "failed")
        return False
    with open(input_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]
    total = len(samples)
    train_ratio = config.get("train_ratio", 0.90)
    val_ratio = config.get("val_ratio", 0.05)
    seed = config.get("split_seed", 42)
    random.seed(seed)
    random.shuffle(samples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    train_samples = samples[:train_size]
    val_samples = samples[train_size : train_size + val_size]
    test_samples = samples[train_size + val_size :]
    images_src = d["prepare"]
    images_dst = output_dir / "images"
    images_dst.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train.jsonl", train_samples), ("val.jsonl", val_samples), ("test.jsonl", test_samples)]:
        filepath = output_dir / split_name
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in split_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        add_log(f"Split: {split_name} = {len(split_data)} samples")
    referenced = {s["image"].replace("images/", "") for s in samples}
    for s in samples:
        img_name = s["image"].replace("images/", "")
        src_img = images_src / img_name
        dst_img = images_dst / img_name
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(str(src_img), str(dst_img))
    stale = [f for f in images_dst.glob("*.png") if f.name not in referenced]
    for f in stale:
        f.unlink()
    if stale:
        add_log(f"Split: removed {len(stale)} stale images no longer in the dataset")
    update_stage(stage_id, processed=1, success=1, extra={"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)})
    finish_stage(stage_id)
    add_log(f"Split complete: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    return True


STAGE_FUNCS = {
    "crop_pii": stage_crop_pii,
    "deskew": stage_deskew,
    "filter_blank": stage_filter_blank,
    "convert_jpg": stage_convert_jpg,
    "prepare": stage_prepare,
    "annotate": stage_annotate,
    "quality_review": stage_quality_review,
    "validate": stage_validate,
    "build": stage_build,
    "split": stage_split,
}


def run_pipeline(config, start_from=None):
    with state_lock:
        pipeline_state["running"] = True
        pipeline_state["stop_requested"] = False
        pipeline_state["paused"] = False
        pipeline_state["started_at"] = time.time()
        pipeline_state["finished_at"] = None
        pipeline_state["config"] = config
        pipeline_state["completed_stages"] = []
        pipeline_state["stages"] = {}
    add_log("=" * 60)
    add_log("PIPELINE STARTED")
    add_log(f"Source: {config.get('source_dir', str(DATA_DIR))}")
    add_log(f"Workspace: {WORK_DIR}")
    add_log("=" * 60)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    stage_ids = [s["id"] for s in PIPELINE_STAGES]
    if start_from and start_from in stage_ids:
        idx = stage_ids.index(start_from)
        stage_ids = stage_ids[idx:]
        add_log(f"Starting from stage: {start_from}")
    all_ok = True
    for sid in stage_ids:
        if pipeline_state["stop_requested"]:
            add_log("Pipeline stopped by user.", "warn")
            all_ok = False
            break
        add_log(f"--- Stage: {sid} ---")
        func = STAGE_FUNCS.get(sid)
        if not func:
            add_log(f"No function for stage {sid}", "error")
            all_ok = False
            break
        try:
            ok = func(config)
        except Exception as e:
            add_log(f"Stage {sid} crashed: {e}", "error")
            finish_stage(sid, "failed")
            ok = False
        if not ok:
            add_log(f"Stage {sid} did not complete successfully.", "warn")
            all_ok = False
            if pipeline_state["stop_requested"]:
                break
    with state_lock:
        pipeline_state["running"] = False
        pipeline_state["finished_at"] = time.time()
    if all_ok:
        add_log("=" * 60)
        add_log("PIPELINE COMPLETED SUCCESSFULLY")
        add_log("=" * 60)
    else:
        add_log("=" * 60)
        add_log("PIPELINE FINISHED (with issues - check logs)")
        add_log("=" * 60)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with state_lock:
        s = json.loads(json.dumps(pipeline_state))
    elapsed = None
    if s.get("started_at"):
        end = s.get("finished_at") or time.time()
        elapsed = round(end - s["started_at"], 1)
    s["elapsed"] = elapsed
    s["stages_meta"] = {st["id"]: st["name"] for st in PIPELINE_STAGES}
    s["stages_order"] = [st["id"] for st in PIPELINE_STAGES]
    return jsonify(s)


@app.route("/api/logs")
def api_logs():
    after = request.args.get("after", 0, type=int)
    with log_lock:
        logs = [e for e in log_buffer if e["id"] > after]
    return jsonify({"logs": logs, "last_id": log_event_id})


@app.route("/api/stream")
def api_stream():
    def generate():
        last_log_id = 0
        last_state_hash = ""
        while True:
            with state_lock:
                state_copy = json.loads(json.dumps(pipeline_state))
            elapsed = None
            if state_copy.get("started_at"):
                end = state_copy.get("finished_at") or time.time()
                elapsed = round(end - state_copy["started_at"], 1)
            state_copy["elapsed"] = elapsed
            state_json = json.dumps(state_copy, sort_keys=True)
            state_hash = hashlib.md5(state_json.encode()).hexdigest()[:12]
            with log_lock:
                new_logs = [e for e in log_buffer if e["id"] > last_log_id]
                if new_logs:
                    last_log_id = new_logs[-1]["id"]
            if state_hash != last_state_hash or new_logs:
                last_state_hash = state_hash
                payload = {"state": state_copy, "new_logs": new_logs}
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/start", methods=["POST"])
def api_start():
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 400
    config = request.json or {}
    if not config.get("api_key"):
        config["api_key"] = os.environ.get("TOKENROUTER_API_KEY", "")
    if not config.get("source_dir"):
        config["source_dir"] = str(DATA_DIR)
    config.setdefault("crop_pct", 10)
    config.setdefault("dark_thresh", 210)
    config.setdefault("max_dark_pct", 0.5)
    config.setdefault("max_size", 2048)
    config.setdefault("api_base_url", "https://api.tokenrouter.com/v1")
    config.setdefault("model", "MiniMax-M3")
    config.setdefault("annotate_delay", 0)
    config.setdefault("max_retries", 3)
    config.setdefault("annotate_workers", 8)
    config.setdefault("max_tokens", 8192)
    config.setdefault("temperature", 0.1)
    config.setdefault("api_key_limit", 0)
    config.setdefault("cpu_workers", default_cpu_workers())
    config.setdefault("validate_workers", 4)
    config.setdefault("train_ratio", 0.90)
    config.setdefault("val_ratio", 0.05)
    config.setdefault("split_seed", 42)
    start_from = config.get("start_from", None)
    t = threading.Thread(target=run_pipeline, args=(config, start_from), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    if not pipeline_state["running"]:
        return jsonify({"error": "Pipeline is not running"}), 400
    with state_lock:
        pipeline_state["stop_requested"] = True
    add_log("Stop requested by user.", "warn")
    return jsonify({"status": "stop_requested"})


@app.route("/api/restart", methods=["POST"])
def api_restart():
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running. Stop it first."}), 400
    
    global log_event_id
    with state_lock:
        pipeline_state["running"] = False
        pipeline_state["paused"] = False
        pipeline_state["stop_requested"] = False
        pipeline_state["current_stage"] = None
        pipeline_state["completed_stages"] = []
        pipeline_state["stages"] = {}
        pipeline_state["started_at"] = None
        pipeline_state["finished_at"] = None
        pipeline_state["config"] = {}
    
    with log_lock:
        log_buffer.clear()
        log_event_id = 0
    
    if WORK_DIR.exists():
        add_log(f"Deleting workspace: {WORK_DIR}", "warn")
        shutil.rmtree(WORK_DIR)
        add_log("Workspace deleted successfully", "info")
    
    add_log("=" * 60)
    add_log("PIPELINE RESTART - Workspace cleared")
    add_log("=" * 60)
    
    return jsonify({"status": "restarted", "workspace_deleted": True})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 400
    
    if not WORK_DIR.exists():
        return jsonify({"error": "No workspace found. Start a new pipeline first."}), 400
    
    config = request.json or {}
    d = dirs()
    source = Path(config.get("source_dir") or str(DATA_DIR))

    src_files = {f.name for f in get_image_files(source)}
    cropped = {f.name for f in get_image_files(d["crop_pii"])}
    kept = {f.name for f in get_image_files(d["filter_blank"])}
    rejected = {f.name for f in get_image_files(d["filter_blank_rejected"])}
    deskewed = {f.name for f in get_image_files(d["deskew"])}
    converted = {f.name for f in d["convert_jpg"].glob("*.png")} if d["convert_jpg"].exists() else set()
    manifest_known = set()
    mp = WORK_DIR / "image_manifest.json"
    if mp.exists():
        try:
            manifest_known = {m["original"] for m in json.loads(mp.read_text(encoding="utf-8"))}
        except Exception:
            manifest_known = set()
    prepared = {f.stem for f in d["prepare"].glob("*.png")} if d["prepare"].exists() else set()
    tex_done = {f.stem for f in d["annotate"].glob("*.tex")} if d["annotate"].exists() else set()

    if not src_files or src_files - cropped:
        resume_stage = "crop_pii"
    elif cropped - (kept | rejected):
        resume_stage = "filter_blank"
    elif kept - deskewed:
        resume_stage = "deskew"
    elif {Path(n).stem + ".png" for n in deskewed} - converted:
        resume_stage = "convert_jpg"
    elif converted - manifest_known:
        resume_stage = "prepare"
    elif prepared - tex_done:
        resume_stage = "annotate"
    elif not d["quality_review"].exists():
        resume_stage = "quality_review"
    elif not d["validate"].exists():
        resume_stage = "validate"
    elif not (d["build"] / "full_dataset.jsonl").exists():
        resume_stage = "build"
    elif not (d["split"] / "train.jsonl").exists():
        resume_stage = "split"
    else:
        resume_stage = None

    if resume_stage is None:
        return jsonify({"error": "Pipeline appears to be complete. Use restart to start over."}), 400
    if not config.get("api_key"):
        config["api_key"] = os.environ.get("TOKENROUTER_API_KEY", "")
    if not config.get("source_dir"):
        config["source_dir"] = str(DATA_DIR)
    config.setdefault("crop_pct", 10)
    config.setdefault("dark_thresh", 210)
    config.setdefault("max_dark_pct", 0.5)
    config.setdefault("max_size", 2048)
    config.setdefault("api_base_url", "https://api.tokenrouter.com/v1")
    config.setdefault("model", "MiniMax-M3")
    config.setdefault("annotate_delay", 0)
    config.setdefault("max_retries", 3)
    config.setdefault("annotate_workers", 8)
    config.setdefault("max_tokens", 8192)
    config.setdefault("temperature", 0.1)
    config.setdefault("api_key_limit", 0)
    config.setdefault("cpu_workers", default_cpu_workers())
    config.setdefault("validate_workers", 4)
    config.setdefault("train_ratio", 0.90)
    config.setdefault("val_ratio", 0.05)
    config.setdefault("split_seed", 42)
    
    add_log("=" * 60)
    add_log(f"PIPELINE RESUME - Starting from stage: {resume_stage}")
    add_log("=" * 60)
    
    t = threading.Thread(target=run_pipeline, args=(config, resume_stage), daemon=True)
    t.start()
    
    return jsonify({"status": "resumed", "resume_stage": resume_stage})


@app.route("/api/scan")
def api_scan():
    source = request.args.get("source", str(DATA_DIR))
    d = Path(source)
    if not d.exists():
        return jsonify({"error": "Source directory not found", "count": 0})
    files = get_image_files(d)
    total_size = sum(f.stat().st_size for f in files)
    return jsonify({"count": len(files), "total_size_mb": round(total_size / 1024 / 1024, 1), "source": str(d)})


@app.route("/api/workspace")
def api_workspace():
    d = dirs()
    result = {}
    for key, path in d.items():
        p = Path(path)
        if p.exists():
            files = list(p.iterdir())
            img_count = len([f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            tex_count = len([f for f in files if f.suffix.lower() == ".tex"])
            jsonl_count = len([f for f in files if f.suffix.lower() == ".jsonl"])
            result[key] = {"path": str(p), "images": img_count, "tex": tex_count, "jsonl": jsonl_count, "total": len(files)}
        else:
            result[key] = {"path": str(p), "images": 0, "tex": 0, "jsonl": 0, "total": 0}
    return jsonify(result)


@app.route("/api/preview")
def api_preview():
    stage = request.args.get("stage", "crop_pii")
    d = dirs()
    path = d.get(stage)
    if not path or not path.exists():
        return jsonify({"images": []})
    files = sorted([f for f in path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:5]
    return jsonify({"images": [f.name for f in files], "path": str(path)})


# ----------------------------------------------------------------------------
# Fine-tuning + benchmark management (runs as subprocesses; metrics via JSON)
# ----------------------------------------------------------------------------
OUTPUT_BASE = BASE_DIR / "output"
TRAIN_SCRIPT = Path(__file__).parent / "train_glm_ocr.py"
BENCH_SCRIPT = Path(__file__).parent / "benchmark_glm_ocr.py"

train_job = {"proc": None, "log": None, "output_dir": None, "started_at": None, "params": {}}
bench_job = {"proc": None, "log": None, "output_dir": None, "started_at": None, "params": {}}
# Baidu Unlimited-OCR fine-tuning is a SEPARATE job in its own venv + folders, so
# it never touches GLM-OCR's models/data.
baidu_train_job = {"proc": None, "log": None, "output_dir": None, "started_at": None, "params": {}}
baidu_ft_job = {"proc": None, "log": None, "output_dir": None, "started_at": None, "params": {}}
scorer_job = {"proc": None, "log": None, "output_dir": None, "started_at": None, "params": {}}
BAIDU_VENV_PY = Path(r"D:\Claude Code\BaiduOCR\venv\Scripts\python.exe")
BAIDU_TRAIN_SCRIPT = Path(r"D:\Claude Code\BaiduOCR\train_unlimited_ocr.py")
BAIDU_FT_GEN_SCRIPT = Path(r"D:\Claude Code\BaiduOCR\gen_baidu_ft_for_benchmark.py")
SCORER_SCRIPT = Path(__file__).parent / "score_models_live.py"
BAIDU_FT_BASE = Path(r"D:\Claude Code\BaiduOCR\finetune")
job_lock = threading.Lock()


def _job_alive(job):
    return job["proc"] is not None and job["proc"].poll() is None


# ── Auto power-off when work finishes (sleep / shut down) ─────────────────────
POWER_FILE = OUTPUT_BASE / "power_action.json"
POWER_GRACE_S = 90  # seconds of continuous idle (after a job ends) before firing
power_state = {"idle_since": None, "saw_work": False, "firing": False, "fired_action": None}


def _read_power_action():
    try:
        if POWER_FILE.exists():
            return json.loads(POWER_FILE.read_text(encoding="utf-8")).get("action") or "none"
    except Exception:
        pass
    return "none"


def _write_power_action(action):
    try:
        POWER_FILE.write_text(json.dumps({"action": action, "armed_at": time.time()}), encoding="utf-8")
    except Exception:
        pass


def _fresh(path, secs=240):
    try:
        return (time.time() - Path(path).stat().st_mtime) < secs
    except Exception:
        return False


def _benchmark_running_now():
    rf = OUTPUT_BASE / "benchmark_results.json"
    try:
        if rf.exists() and _fresh(rf):
            return json.loads(rf.read_text(encoding="utf-8")).get("state") == "running"
    except Exception:
        pass
    return False


def _baidu_gen_running():
    pf = OUTPUT_BASE / "baidu_gen_progress.json"
    try:
        if pf.exists() and _fresh(pf):
            return json.loads(pf.read_text(encoding="utf-8")).get("status") != "done"
    except Exception:
        pass
    return False


def _baidu_ft_gen_running():
    # the FT benchmark may run as a Scheduled Task (untracked job), so detect it via
    # a fresh, not-done progress file — otherwise it would collide with training
    pf = OUTPUT_BASE / "baidu_ft_gen_progress.json"
    try:
        if pf.exists() and _fresh(pf):
            return json.loads(pf.read_text(encoding="utf-8")).get("status") not in ("done", "stopped")
    except Exception:
        pass
    return False


def _training_running():
    if _job_alive(train_job):
        return True
    try:
        _, alive = _find_detached_run()
        return bool(alive)
    except Exception:
        return False


def _any_work_running():
    """True if any benchmark, training, the data pipeline, or the Baidu gen/scoring
    is still working — the auto power-off only fires once everything is idle."""
    try:
        if pipeline_state.get("running"):
            return True
    except Exception:
        pass
    return (_job_alive(bench_job) or _training_running()
            or _baidu_training_running() or _baidu_ft_gen_running()
            or _benchmark_running_now() or _baidu_gen_running())


def _fire_power(action):
    add_log(f"Auto power-off: all work finished — executing '{action}'.", "warn")
    power_state["firing"] = True
    power_state["fired_action"] = action
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        if action == "shutdown":
            subprocess.Popen(["shutdown", "/s", "/t", "60", "/c",
                              "OCR2TeX: work finished - shutting down in 60s. Cancel in the dashboard or run: shutdown /a"],
                             creationflags=flags)
        elif action == "sleep":
            subprocess.Popen(["powershell", "-NoProfile", "-Command",
                              "Add-Type -AssemblyName System.Windows.Forms; "
                              "[System.Windows.Forms.Application]::SetSuspendState("
                              "[System.Windows.Forms.PowerState]::Suspend,$false,$false)"],
                             creationflags=flags)
    except Exception as e:
        add_log(f"Auto power-off failed to run {action}: {e}", "error")


def _power_watcher():
    """Background daemon: once armed, fire the chosen action after a job that was
    running finishes and the machine stays idle for POWER_GRACE_S."""
    while True:
        try:
            action = _read_power_action()
            if action in ("sleep", "shutdown"):
                if _any_work_running():
                    power_state["saw_work"] = True
                    power_state["idle_since"] = None
                elif power_state["saw_work"]:  # only fire after work was seen then ended
                    if power_state["idle_since"] is None:
                        power_state["idle_since"] = time.time()
                    elif time.time() - power_state["idle_since"] >= POWER_GRACE_S:
                        _fire_power(action)
                        _write_power_action("none")  # one-shot; disarm after firing
                        power_state["idle_since"] = None
                        power_state["saw_work"] = False
            else:
                power_state["idle_since"] = None
        except Exception:
            pass
        time.sleep(15)


def model_label(spec):
    if spec == "base":
        return "Base GLM-OCR"
    p = Path(spec)
    return p.parent.name if p.name == "final" else p.name


def _tail_file(path, lines=60):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return text.splitlines()[-lines:]
    except Exception:
        return []


def _spawn_job(job, cmd, log_path):
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    job["proc"] = proc
    job["log"] = str(log_path)
    job["started_at"] = time.time()
    return proc


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    with job_lock:
        if _job_alive(train_job):
            return jsonify({"error": "Training is already running"}), 400
        _, detached_alive = _find_detached_run()
        if train_job["output_dir"] is None and detached_alive:
            return jsonify({"error": "A training run is already active (it survived a dashboard restart). Wait for it to finish."}), 400
        if _job_alive(bench_job):
            return jsonify({"error": "Benchmark is running - stop it first (GPU is busy)"}), 400
        p = request.json or {}
        run_name = str(p.get("run_name", "glm-ocr-math-v4")).strip() or "glm-ocr-math-v4"
        output_dir = OUTPUT_BASE / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", str(TRAIN_SCRIPT),
            "--output-dir", str(output_dir),
            "--epochs", str(p.get("epochs", 2)),
            "--batch-size", str(p.get("batch_size", 1)),
            "--grad-accum", str(p.get("grad_accum", 8)),
            "--lr", str(p.get("lr", 2e-5)),
            "--max-length", str(p.get("max_length", 3584)),
            "--lora-r", str(p.get("lora_r", 32)),
            "--lora-alpha", str(p.get("lora_alpha", 64)),
            "--lora-dropout", str(p.get("lora_dropout", 0.05)),
            "--max-image-tokens", str(p.get("max_image_tokens", 1536)),
            "--eval-samples", str(p.get("eval_samples", 200)),
            "--early-stopping-patience", str(p.get("early_stopping_patience", 3)),
            "--workers", str(p.get("workers", 2)),
            "--seed", str(p.get("seed", 42)),
            "--max-steps", str(p.get("max_steps", -1)),
        ]
        if p.get("resume"):
            cmd.append("--resume")
        train_job["params"] = p
        train_job["output_dir"] = str(output_dir)
        _spawn_job(train_job, cmd, output_dir / "train_log.txt")
        add_log(f"Training started: {run_name} (pid {train_job['proc'].pid})")
        return jsonify({"status": "started", "output_dir": str(output_dir)})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    with job_lock:
        if not _job_alive(train_job):
            return jsonify({"error": "Training is not running"}), 400
        train_job["proc"].terminate()
        add_log("Training stop requested (checkpoints are preserved)", "warn")
        return jsonify({"status": "stopping"})


# ── Baidu Unlimited-OCR fine-tuning (separate venv, separate folders) ─────────
def _baidu_training_running():
    """True if a Baidu fine-tune is active — including one orphaned by a dashboard
    restart (detected via a fresh training_metrics.json), so the guard can't be
    fooled into starting a second GPU job."""
    if _job_alive(baidu_train_job):
        return True
    try:
        for mf in BAIDU_FT_BASE.glob("*/training_metrics.json"):
            d = json.loads(mf.read_text(encoding="utf-8"))
            if d.get("status") in ("running", "loading") and _fresh(mf, 420):
                return True
    except Exception:
        pass
    return False


def _kill_py_scripts(names):
    """Force-kill any python process whose script basename is in `names` — catch-all
    for jobs launched as scheduled tasks / detached, not just tracked subprocesses."""
    try:
        import psutil
    except Exception:
        return
    for pr in psutil.process_iter(["cmdline"]):
        try:
            scripts = [a for a in (pr.info["cmdline"] or []) if a.lower().endswith(".py")]
            if scripts and os.path.basename(scripts[-1]) in names:
                pr.kill()
        except Exception:
            pass


def _py_script_alive(name):
    try:
        import psutil
        for pr in psutil.process_iter(["cmdline"]):
            try:
                s = [a for a in (pr.info["cmdline"] or []) if a.lower().endswith(".py")]
                if s and os.path.basename(s[-1]) == name:
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _end_task(name):
    try:
        subprocess.run(["schtasks", "/end", "/tn", name], capture_output=True, timeout=10)
    except Exception:
        pass


def _stop_baidu_training():
    """Stop a Baidu fine-tune however it was launched (subprocess / scheduled task /
    detached). Checkpoints are preserved; metrics are marked stopped and we wait for
    the process to actually exit so the GPU frees before anything else grabs it."""
    try:
        if _job_alive(baidu_train_job):
            baidu_train_job["proc"].terminate()
    except Exception:
        pass
    _end_task("BaiduFineTune")
    _kill_py_scripts({"train_unlimited_ocr.py"})
    try:
        for mf in BAIDU_FT_BASE.glob("*/training_metrics.json"):
            d = json.loads(mf.read_text(encoding="utf-8"))
            if d.get("status") in ("running", "loading"):
                d["status"] = "stopped"
                mf.write_text(json.dumps(d), encoding="utf-8")
    except Exception:
        pass
    for _ in range(20):  # let the GPU free before returning
        if not _py_script_alive("train_unlimited_ocr.py"):
            break
        time.sleep(0.3)


def _stop_ft_benchmark():
    try:
        if _job_alive(baidu_ft_job):
            baidu_ft_job["proc"].terminate()
    except Exception:
        pass
    _end_task("OCR2TeXFTgen")
    _kill_py_scripts({"gen_baidu_ft_for_benchmark.py"})
    try:
        pf = OUTPUT_BASE / "baidu_ft_gen_progress.json"
        if pf.exists():
            d = json.loads(pf.read_text(encoding="utf-8"))
            if d.get("status") != "done":
                d["status"] = "stopped"
                pf.write_text(json.dumps(d), encoding="utf-8")
    except Exception:
        pass


def _gpu_busy_reason():
    """Anything currently using the single GPU — block a new Baidu GPU job."""
    if _job_alive(bench_job):
        return "the benchmark is running"
    if _job_alive(train_job):
        return "GLM-OCR training is running"
    if _baidu_training_running():
        return "Baidu training is running"
    if _job_alive(baidu_ft_job):
        return "the fine-tuned-Baidu benchmark is running"
    if _baidu_ft_gen_running():
        return "the fine-tuned-Baidu benchmark is running"
    if _baidu_gen_running():
        return "the Baidu benchmark generation is running"
    return None


@app.route("/api/baidu_train/start", methods=["POST"])
def api_baidu_train_start():
    with job_lock:
        busy = _gpu_busy_reason()
        if busy:
            return jsonify({"error": f"GPU is busy — {busy}. Free it first."}), 400
        if not BAIDU_TRAIN_SCRIPT.exists():
            return jsonify({"error": "Baidu trainer not found"}), 400
        p = request.json or {}
        run_name = str(p.get("run_name", "baidu-ocr-math-v1")).strip() or "baidu-ocr-math-v1"
        output_dir = BAIDU_FT_BASE / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(BAIDU_VENV_PY), "-u", str(BAIDU_TRAIN_SCRIPT),
            "--output-dir", str(output_dir),
            "--epochs", str(p.get("epochs", 1)),
            "--lr", str(p.get("lr", 1e-4)),
            "--lora-r", str(p.get("lora_r", 16)),
            "--lora-alpha", str(p.get("lora_alpha", 32)),
            "--lora-dropout", str(p.get("lora_dropout", 0.05)),
            "--grad-accum", str(p.get("grad_accum", 8)),
            "--max-target-len", str(p.get("max_target_len", 1024)),
            "--eval-samples", str(p.get("eval_samples", 80)),
            "--save-steps", str(p.get("save_steps", 200)),
            "--eval-steps", str(p.get("eval_steps", 200)),
            "--early-stopping-patience", str(p.get("early_stopping_patience", 0)),
            "--max-steps", str(p.get("max_steps", 0)),
            "--max-crops", str(p.get("max_crops", 9)),
        ]
        if p.get("qlora"):
            cmd.append("--qlora")
        if p.get("resume"):
            cmd.append("--resume")
        baidu_train_job["params"] = p
        baidu_train_job["output_dir"] = str(output_dir)
        _spawn_job(baidu_train_job, cmd, output_dir / "train_log.txt")
        add_log(f"Baidu fine-tune started: {run_name} (pid {baidu_train_job['proc'].pid})")
        return jsonify({"status": "started", "output_dir": str(output_dir)})


@app.route("/api/baidu_train/stop", methods=["POST"])
def api_baidu_train_stop():
    with job_lock:
        if not _baidu_training_running():
            return jsonify({"error": "Baidu training is not running"}), 400
        _stop_baidu_training()
        add_log("Baidu fine-tune stopped (checkpoint preserved — use Resume to continue)", "warn")
        return jsonify({"status": "stopping"})


@app.route("/api/baidu_train/state")
def api_baidu_train_state():
    running = _job_alive(baidu_train_job)
    output_dir = baidu_train_job["output_dir"]
    out = {"running": running, "output_dir": output_dir, "params": baidu_train_job["params"],
           "gpu_busy": _gpu_busy_reason(), "metrics": None, "log_tail": []}
    # recover a detached run from the newest metrics file under the Baidu folder
    if not output_dir and BAIDU_FT_BASE.exists():
        cands = sorted(BAIDU_FT_BASE.glob("*/training_metrics.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if cands:
            output_dir = str(cands[0].parent)
    if output_dir:
        mf = Path(output_dir) / "training_metrics.json"
        if mf.exists():
            try:
                out["metrics"] = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                pass
            # a run launched in a prior session survives as an orphan; show it as
            # running if its metrics say so and are fresh (so a dashboard restart
            # doesn't make an overnight run look stopped)
            if not running and (out["metrics"] or {}).get("status") in ("running", "loading") and _fresh(mf, 420):
                out["running"] = True
                out["detached"] = True
        out["log_tail"] = _tail_file(baidu_train_job["log"] or (Path(output_dir) / "train_log.txt"), 60)
    return jsonify(out)


def _ensure_scorer():
    """Make sure the unified scorer is running (scores Baidu + Baidu-FT columns)."""
    if not _job_alive(scorer_job):
        _spawn_job(scorer_job, [sys.executable, "-u", str(SCORER_SCRIPT)], OUTPUT_BASE / "score_models_log.txt")


@app.route("/api/baidu_ft_bench/start", methods=["POST"])
def api_baidu_ft_bench_start():
    """Benchmark the FINE-TUNED Baidu model: generate on the test set (base+adapter)
    and score it into a new 'Baidu OCR FT' column, same as every other model."""
    with job_lock:
        # Starting the benchmark intentionally STOPS a running fine-tune (single GPU);
        # the checkpoint is preserved so you can Resume training afterward.
        stopped_training = False
        if _baidu_training_running():
            _stop_baidu_training()
            stopped_training = True
            add_log("Stopped Baidu training to run the FT benchmark (checkpoint preserved).", "warn")
        # still refuse to fight GLM jobs (those we don't auto-kill)
        if _job_alive(bench_job):
            return jsonify({"error": "GPU is busy — the GLM benchmark is running. Stop it first."}), 400
        if _job_alive(train_job):
            return jsonify({"error": "GPU is busy — GLM-OCR training is running. Stop it first."}), 400
        if _job_alive(baidu_ft_job) or _baidu_ft_gen_running():
            return jsonify({"error": "The fine-tuned-Baidu benchmark is already running."}), 400
        p = request.json or {}
        run = str(p.get("run_name", "baidu-ocr-math-v1")).strip() or "baidu-ocr-math-v1"
        rdir = BAIDU_FT_BASE / run
        if not any((rdir / sub / "adapter_config.json").exists() for sub in ("best", "final", ".")):
            return jsonify({"error": f"No trained adapter in {rdir} — finish or Resume training first."}), 400
        samples = int(p.get("samples", 700))
        cmd = [str(BAIDU_VENV_PY), "-u", str(BAIDU_FT_GEN_SCRIPT), "--samples", str(samples)]
        if p.get("adapter"):
            cmd += ["--adapter", str(p["adapter"])]
        baidu_ft_job["params"] = {"run_name": run, "samples": samples}
        _spawn_job(baidu_ft_job, cmd, OUTPUT_BASE / "baidu_ft_gen_log.txt")
        _ensure_scorer()
        add_log(f"Fine-tuned-Baidu benchmark started ({samples} pages; adapter '{run}')")
        return jsonify({"status": "started", "samples": samples, "stopped_training": stopped_training})


@app.route("/api/baidu_ft_bench/stop", methods=["POST"])
def api_baidu_ft_bench_stop():
    with job_lock:
        if not (_job_alive(baidu_ft_job) or _baidu_ft_gen_running()):
            return jsonify({"error": "Fine-tuned-Baidu benchmark is not running"}), 400
        _stop_ft_benchmark()
        add_log("Fine-tuned-Baidu benchmark stopped (resumable).", "warn")
        return jsonify({"status": "stopping"})


def _find_detached_run():
    """After a dashboard restart the training subprocess keeps running but the
    process handle is lost. Recover the most recently active run from its
    metrics file so the UI keeps showing live state."""
    newest = None
    if OUTPUT_BASE.exists():
        for mf in OUTPUT_BASE.glob("*/training_metrics.json"):
            if newest is None or mf.stat().st_mtime > newest.stat().st_mtime:
                newest = mf
    if newest is None:
        return None, False
    age = time.time() - newest.stat().st_mtime
    try:
        state = json.loads(newest.read_text(encoding="utf-8")).get("state")
    except Exception:
        state = None
    # metrics update at least every logging step (~1 min); 5 min stale = dead
    alive = state == "running" and age < 300
    return newest.parent, alive


@app.route("/api/train/state")
def api_train_state():
    detached = False
    output_dir = train_job["output_dir"]
    running = _job_alive(train_job)
    if not running and output_dir is None:
        recovered_dir, alive = _find_detached_run()
        if recovered_dir is not None:
            output_dir = str(recovered_dir)
            running = alive
            detached = alive
    out = {
        "running": running,
        "detached": detached,
        "returncode": train_job["proc"].poll() if train_job["proc"] else None,
        "output_dir": output_dir,
        "params": train_job["params"],
        "started_at": train_job["started_at"],
        "metrics": None,
        "log_tail": [],
    }
    if output_dir:
        mf = Path(output_dir) / "training_metrics.json"
        if mf.exists():
            try:
                out["metrics"] = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                pass
        log_path = train_job["log"] or (Path(output_dir) / "train_log.txt")
        out["log_tail"] = _tail_file(log_path, 80)
    return jsonify(out)


@app.route("/api/maintenance/requeue", methods=["POST"])
def api_maintenance_requeue():
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline is running - stop it first"}), 400
    d = dirs()
    rej_dir = d["validate_rejected"]
    ann_dir = d["annotate"]
    count = 0
    if rej_dir.exists():
        for f in rej_dir.glob("*.tex"):
            target = ann_dir / f.name
            if target.exists():
                target.unlink()
                count += 1
    add_log(f"Maintenance: requeued {count} rejected pages for re-annotation. "
            f"Set Temperature (and Quarantine if desired) then click Resume.", "warn")
    return jsonify({"status": "ok", "requeued": count})


@app.route("/api/maintenance/build_salvaged", methods=["POST"])
def api_maintenance_build_salvaged():
    script = Path(__file__).parent / "build_salvaged_dataset.py"
    def run():
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           cwd=str(Path(__file__).parent))
        for line in (r.stdout or "").strip().splitlines():
            add_log(f"Salvaged dataset: {line}")
        if r.returncode != 0:
            add_log(f"Salvaged dataset build failed: {(r.stderr or '')[-200:]}", "error")
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/benchmark/start", methods=["POST"])
def api_benchmark_start():
    with job_lock:
        if _job_alive(bench_job):
            return jsonify({"error": "Benchmark is already running"}), 400
        if _job_alive(train_job):
            return jsonify({"error": "Training is running - benchmark would fight for the GPU"}), 400
        p = request.json or {}
        # accept either a "models" list or legacy model_a/b/c keys
        models = p.get("models")
        if not models:
            models = [p.get("model_a") or "base", p.get("model_b") or str(OUTPUT_BASE / "glm-ocr-math-v4" / "final")]
            if p.get("model_c") and p["model_c"] not in ("none", ""):
                models.append(p["model_c"])
        models = [m for m in models if m and m != "none"]
        if len(models) < 2:
            return jsonify({"error": "Pick at least 2 models to compare"}), 400
        if len(set(models)) != len(models):
            return jsonify({"error": "Models must be distinct"}), 400
        for spec in models:
            if spec != "base" and not Path(spec).exists():
                return jsonify({"error": f"Adapter not found: {spec}"}), 400
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", str(BENCH_SCRIPT),
            "--models", *models,
            "--samples", str(p.get("samples", 100)),
            "--max-new-tokens", str(p.get("max_new_tokens", 2048)),
            "--rep-penalty", str(p.get("rep_penalty", 1.0)),
            "--no-repeat-ngram", str(p.get("no_repeat_ngram", 0)),
            "--batch-size", str(p.get("batch_size", 4)),
            "--output", str(OUTPUT_BASE / "benchmark_results.json"),
        ]
        if p.get("fresh"):
            cmd.append("--fresh")
        bench_job["params"] = p
        bench_job["output_dir"] = str(OUTPUT_BASE)
        _spawn_job(bench_job, cmd, OUTPUT_BASE / "benchmark_log.txt")
        add_log(f"Benchmark started: {' vs '.join(model_label(m) for m in models)} (pid {bench_job['proc'].pid})")
        return jsonify({"status": "started"})


ADAPTER_SCAN_DIRS = [OUTPUT_BASE, Path(r"D:\Kiro\output")]


@app.route("/api/adapters")
def api_adapters():
    found = []
    for base in ADAPTER_SCAN_DIRS:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            final = d / "final"
            if (final / "adapter_config.json").exists():
                found.append({"label": d.name, "path": str(final)})
            # also pick up checkpoints saved directly with adapter_config.json
            elif (d / "adapter_config.json").exists():
                found.append({"label": d.name, "path": str(d)})
    return jsonify({"adapters": found})


@app.route("/api/benchmark/add_model", methods=["POST"])
def api_benchmark_add_model():
    """Add one model to the existing benchmark results without re-running the
    others. The existing models are fully cached, so they're reused instantly;
    only the new model is generated. Same test set / settings as the last run."""
    with job_lock:
        if _job_alive(bench_job):
            return jsonify({"error": "Benchmark is already running"}), 400
        if _job_alive(train_job):
            return jsonify({"error": "Training is running - free the GPU first"}), 400
        p = request.json or {}
        spec = (p.get("model") or "").strip()
        if not spec:
            return jsonify({"error": "Pick a model to add"}), 400
        if spec != "base" and not Path(spec).exists():
            return jsonify({"error": f"Adapter not found: {spec}"}), 400
        rf = OUTPUT_BASE / "benchmark_results.json"
        existing, samples, max_new = [], 100, 2048
        if rf.exists():
            try:
                r = json.loads(rf.read_text(encoding="utf-8"))
                existing = [r["models"][k].get("spec") for k in r.get("model_keys", []) if r["models"][k].get("spec")]
                samples = r.get("samples", 100)
            except Exception:
                pass
        models = list(existing)
        if spec not in models:
            models.append(spec)
        # match the last full run's settings so the new column is comparable
        cmd = [
            sys.executable, "-u", str(BENCH_SCRIPT),
            "--models", *models,
            "--samples", str(samples),
            "--max-new-tokens", "1536",
            "--batch-size", "4",
            "--output", str(rf),
        ]
        bench_job["params"] = {"add_model": spec, "samples": samples}
        bench_job["output_dir"] = str(OUTPUT_BASE)
        _spawn_job(bench_job, cmd, OUTPUT_BASE / "benchmark_log.txt")
        add_log(f"Benchmark: adding {model_label(spec)} (others reused from cache; {samples} samples)")
        return jsonify({"status": "started", "label": model_label(spec), "samples": samples})


@app.route("/api/benchmark/stop", methods=["POST"])
def api_benchmark_stop():
    with job_lock:
        if not _job_alive(bench_job):
            return jsonify({"error": "Benchmark is not running"}), 400
        bench_job["proc"].terminate()
        return jsonify({"status": "stopping"})


@app.route("/api/benchmark/state")
def api_benchmark_state():
    running = _job_alive(bench_job)
    # results file is shared whether the benchmark was launched from the dashboard
    # or from a command line, so always read it from the canonical location
    rf = Path(bench_job["output_dir"] or OUTPUT_BASE) / "benchmark_results.json"
    out = {
        "running": running,
        "returncode": bench_job["proc"].poll() if bench_job["proc"] else None,
        "results": None,
        "log_tail": [],
    }
    if rf.exists():
        try:
            res = json.loads(rf.read_text(encoding="utf-8"))
            # the dashboard only needs summaries/progress; drop the multi-MB
            # per-page rows so we don't ship them every poll
            res.pop("per_sample", None)
            out["results"] = res
        except Exception:
            pass
    # a CLI run isn't a tracked subprocess; treat a freshly-updated results file
    # whose state is still "running" as a live (external) benchmark
    if not running and out["results"] and out["results"].get("state") == "running":
        try:
            if time.time() - rf.stat().st_mtime < 180:
                out["running"] = True
                out["external"] = True
        except Exception:
            pass
    # the benchmark mirrors its own output to benchmark_run.log, so the dashboard
    # can tail it whether the run was launched here or from a command line
    run_log = Path(bench_job["output_dir"] or OUTPUT_BASE) / "benchmark_run.log"
    if run_log.exists():
        out["log_tail"] = _tail_file(run_log, 40)
    elif out.get("external"):
        out["log_tail"] = [
            "Running from a command-line terminal — live progress is shown above.",
            "(This run predates per-page log capture; detailed output is in your terminal.)",
        ]
    else:
        log_path = bench_job["log"] or (OUTPUT_BASE / "benchmark_log.txt")
        out["log_tail"] = _tail_file(log_path, 40)
    return jsonify(out)


@app.route("/api/system")
def api_system():
    try:
        import psutil
        # interval=None returns 0.0 on the first call (no baseline); a short
        # blocking sample gives a real reading on every request
        cpu = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        ram_used = round(vm.used / 1024**3, 1)
        ram_total = round(vm.total / 1024**3, 1)
    except Exception:
        cpu, ram_used, ram_total = None, None, None
    gpu = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
            gpu = {
                "util_pct": float(parts[0]),
                "mem_used_mb": float(parts[1]),
                "mem_total_mb": float(parts[2]),
                "temp_c": float(parts[3]),
                "power_w": float(parts[4]),
                "name": parts[5],
            }
    except Exception:
        pass
    return jsonify({"cpu_pct": cpu, "ram_used_gb": ram_used, "ram_total_gb": ram_total, "gpu": gpu})


# ----------------------------------------------------------------------------
# Inspector: view source image + each model's .tex/.pdf, and run new images
# ----------------------------------------------------------------------------
BENCH_OUT = OUTPUT_BASE / "bench_outputs"
INSPECT_DIR = OUTPUT_BASE / "inspect"
TEST_JSONL = WORK_DIR / "9_split" / "test.jsonl"
TEST_IMG_DIR = WORK_DIR / "9_split" / "images"
INFER_SCRIPT = Path(__file__).parent / "infer_glm_ocr.py"

_ref_cache = {}
inspect_job = {"proc": None, "log": None, "uid": None, "out_dir": None}


def _safe_id(s):
    return bool(s) and re.fullmatch(r"[A-Za-z0-9_.\-]+", s) is not None and ".." not in s


def _load_refs():
    if _ref_cache or not TEST_JSONL.exists():
        return _ref_cache
    for line in TEST_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            s = json.loads(line)
            ref = next((c["content"] for c in s["conversations"] if c["role"] == "assistant"), "")
            _ref_cache[s["id"]] = ref
    return _ref_cache


def _model_dirs():
    if not BENCH_OUT.exists():
        return []
    return sorted(d.name for d in BENCH_OUT.iterdir() if d.is_dir() and d.name != "_reference")


@app.route("/api/inspect/config")
def api_inspect_config():
    models = _model_dirs()
    # benchmarked test pages = those with at least one model .tex
    pages = set()
    for m in models:
        for f in (BENCH_OUT / m).glob("*.tex"):
            pages.add(f.stem)
    uploads = []
    if INSPECT_DIR.exists():
        # only true uploads; re-runs overwrite their page in place (no separate entry)
        uploads = sorted((d.name for d in INSPECT_DIR.iterdir() if d.is_dir() and d.name.startswith("upload_")), reverse=True)
    return jsonify({"models": models, "pages": sorted(pages), "uploads": uploads})


@app.route("/api/inspect/page/<pid>")
def api_inspect_page(pid):
    if not _safe_id(pid):
        abort(400)
    out = {"id": pid, "models": {}, "reference": None, "is_upload": (INSPECT_DIR / pid).is_dir()}
    if out["is_upload"]:
        d = INSPECT_DIR / pid
        if not d.exists():
            abort(404)
        latencies = {}
        rj = d / "result.json"
        if rj.exists():
            try:
                res = json.loads(rj.read_text(encoding="utf-8"))
                out["has_handwriting"] = res.get("has_handwriting")
                out["prefiltered"] = res.get("prefiltered", False)
                latencies = {k: v.get("latency_s") for k, v in (res.get("models") or {}).items()}
            except Exception:
                pass
        for tex in sorted(d.glob("*.tex")):
            out["models"][tex.stem] = {"tex": tex.read_text(encoding="utf-8", errors="replace"),
                                       "pdf": (d / f"{tex.stem}.pdf").exists(),
                                       "latency_s": latencies.get(tex.stem)}
    else:
        out["reference"] = _load_refs().get(pid)
        for m in _model_dirs():
            tex = BENCH_OUT / m / f"{pid}.tex"
            if tex.exists():
                out["models"][m] = {"tex": tex.read_text(encoding="utf-8", errors="replace"),
                                    "pdf": (BENCH_OUT / m / f"{pid}.pdf").exists()}
    return jsonify(out)


@app.route("/api/inspect/image/<pid>")
def api_inspect_image(pid):
    if not _safe_id(pid):
        abort(400)
    if (INSPECT_DIR / pid).is_dir():
        p = INSPECT_DIR / pid / "image.png"
    else:
        p = TEST_IMG_DIR / f"{pid}.png"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/png")


@app.route("/api/inspect/pdf/<model>/<pid>")
def api_inspect_pdf(model, pid):
    if not (_safe_id(pid) and _safe_id(model)):
        abort(400)
    if (INSPECT_DIR / pid).is_dir():
        p = INSPECT_DIR / pid / f"{model}.pdf"
    else:
        p = BENCH_OUT / model / f"{pid}.pdf"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="application/pdf")


@app.route("/api/inspect/refpdf/<pid>")
def api_inspect_refpdf(pid):
    if not _safe_id(pid) or (INSPECT_DIR / pid).is_dir():
        abort(404)
    refdir = BENCH_OUT / "_reference"
    refdir.mkdir(parents=True, exist_ok=True)
    pdf = refdir / f"{pid}.pdf"
    if not pdf.exists():
        ref = _load_refs().get(pid)
        if not ref:
            abort(404)
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_glm_ocr import compile_tex
        ok, _ = compile_tex(ref, pdf)
        if not ok:
            abort(404)
    return send_file(str(pdf), mimetype="application/pdf")


def _pdf_to_png(pdf_path, png_path, zoom=2.0):
    import fitz
    doc = fitz.open(str(pdf_path))
    try:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(str(png_path))
    finally:
        doc.close()


@app.route("/api/inspect/pdfimg/<model>/<pid>")
def api_inspect_pdfimg(model, pid):
    # serve a model's compiled PDF as a PNG image (renders inline; never downloads)
    if not (_safe_id(pid) and _safe_id(model)):
        abort(400)
    if (INSPECT_DIR / pid).is_dir():
        pdf, png = INSPECT_DIR / pid / f"{model}.pdf", INSPECT_DIR / pid / f"{model}.render.png"
    else:
        pdf, png = BENCH_OUT / model / f"{pid}.pdf", BENCH_OUT / model / f"{pid}.render.png"
    if not pdf.exists():
        abort(404)
    if (not png.exists()) or png.stat().st_mtime < pdf.stat().st_mtime:
        try:
            _pdf_to_png(pdf, png)
        except Exception:
            abort(404)
    return send_file(str(png), mimetype="image/png")


@app.route("/api/inspect/refpdfimg/<pid>")
def api_inspect_refpdfimg(pid):
    if not _safe_id(pid) or (INSPECT_DIR / pid).is_dir():
        abort(404)
    refdir = BENCH_OUT / "_reference"
    refdir.mkdir(parents=True, exist_ok=True)
    pdf, png = refdir / f"{pid}.pdf", refdir / f"{pid}.render.png"
    if not pdf.exists():
        ref = _load_refs().get(pid)
        if not ref:
            abort(404)
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_glm_ocr import compile_tex
        ok, _ = compile_tex(ref, pdf)
        if not ok:
            abort(404)
    if (not png.exists()) or png.stat().st_mtime < pdf.stat().st_mtime:
        try:
            _pdf_to_png(pdf, png)
        except Exception:
            abort(404)
    return send_file(str(png), mimetype="image/png")


@app.route("/api/inspect/upload", methods=["POST"])
def api_inspect_upload():
    with job_lock:
        if _job_alive(inspect_job):
            return jsonify({"error": "An image is already being processed"}), 400
        if _job_alive(bench_job) or _job_alive(train_job):
            return jsonify({"error": "GPU is busy (benchmark or training running). Try again when it finishes."}), 400
        f = request.files.get("image")
        if not f:
            return jsonify({"error": "No image uploaded"}), 400
        rep = request.form.get("rep_penalty", "1.0")
        nrn = request.form.get("no_repeat_ngram", "0")
        uid = "upload_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        d = INSPECT_DIR / uid
        d.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(f.stream).convert("RGB")
            img.save(d / "image.png")
        except Exception as e:
            return jsonify({"error": f"Bad image: {e}"}), 400
        models = ["base", str(Path(r"D:\Kiro\output\glm-ocr-math-v3.1\final")),
                  str(OUTPUT_BASE / "glm-ocr-math-v4" / "final")]
        models = [m for m in models if m == "base" or Path(m).exists()]
        cmd = [sys.executable, "-u", str(INFER_SCRIPT), "--image", str(d / "image.png"),
               "--out", str(d), "--rep-penalty", rep, "--no-repeat-ngram", nrn, "--models", *models]
        inspect_job["uid"] = uid
        inspect_job["out_dir"] = str(d)
        _spawn_job(inspect_job, cmd, d / "infer_log.txt")
        return jsonify({"status": "started", "uid": uid})


@app.route("/api/inspect/rerun", methods=["POST"])
def api_inspect_rerun():
    with job_lock:
        if _job_alive(inspect_job):
            return jsonify({"error": "An image is already being processed"}), 400
        if _job_alive(bench_job) or _job_alive(train_job):
            return jsonify({"error": "GPU is busy (benchmark or training running)."}), 400
        p = request.json or {}
        pid = p.get("pid", "")
        if not _safe_id(pid):
            return jsonify({"error": "bad page id"}), 400
        # source image: a test page, or an existing inspect (upload/rerun) entry
        if (TEST_IMG_DIR / f"{pid}.png").exists():
            src = TEST_IMG_DIR / f"{pid}.png"
        elif (INSPECT_DIR / pid / "image.png").exists():
            src = INSPECT_DIR / pid / "image.png"
        else:
            return jsonify({"error": "image not found"}), 404
        spec_map = {
            "base": "base",
            "v3.1": str(Path(r"D:\Kiro\output\glm-ocr-math-v3.1\final")),
            "v4": str(OUTPUT_BASE / "glm-ocr-math-v4" / "final"),
            "v4.1": str(OUTPUT_BASE / "glm-ocr-math-v4.1" / "final"),
        }
        sel = [m for m in (p.get("models") or ["v4"]) if m in spec_map]
        specs = [spec_map[m] for m in sel if spec_map[m] == "base" or Path(spec_map[m]).exists()]
        if not specs:
            return jsonify({"error": "no valid model selected"}), 400
        cmd = [sys.executable, "-u", str(INFER_SCRIPT), "--image", str(src),
               "--rep-penalty", str(p.get("rep_penalty", 1.0)),
               "--no-repeat-ngram", str(p.get("no_repeat_ngram", 0)), "--models", *specs]
        if p.get("no_prefilter"):
            cmd.append("--no-prefilter")
        is_test_page = (TEST_IMG_DIR / f"{pid}.png").exists()
        if is_test_page:
            # overwrite the stored benchmark outputs for this page in place
            status_dir = INSPECT_DIR / "_rerun_status"
            status_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--out", str(status_dir), "--bench-out", str(BENCH_OUT), "--page-id", pid]
            out_dir = status_dir
        else:
            # an existing upload/rerun entry: overwrite its files in place
            out_dir = INSPECT_DIR / pid
            cmd += ["--out", str(out_dir)]
        inspect_job["uid"] = pid
        inspect_job["out_dir"] = str(out_dir)
        _spawn_job(inspect_job, cmd, out_dir / "infer_log.txt")
        add_log(f"Inspector re-run: {pid} with {sel} (pid {inspect_job['proc'].pid})")
        return jsonify({"status": "started", "reopen": pid})


@app.route("/api/inspect/upload_state")
def api_inspect_upload_state():
    out = {"running": _job_alive(inspect_job), "uid": inspect_job["uid"], "result": None, "log_tail": []}
    if inspect_job["out_dir"]:
        rf = Path(inspect_job["out_dir"]) / "result.json"
        if rf.exists():
            try:
                out["result"] = json.loads(rf.read_text(encoding="utf-8"))
            except Exception:
                pass
        if inspect_job["log"]:
            out["log_tail"] = _tail_file(inspect_job["log"], 20)
    return jsonify(out)


@app.route("/api/power/state")
def api_power_state():
    action = _read_power_action()
    working = _any_work_running()
    idle = power_state["idle_since"]
    countdown = None
    if action in ("sleep", "shutdown") and power_state["saw_work"] and not working and idle:
        countdown = max(0, POWER_GRACE_S - int(time.time() - idle))
    return jsonify({
        "action": action,
        "work_running": working,
        "saw_work": power_state["saw_work"],
        "firing": power_state["firing"],
        "fired_action": power_state["fired_action"],
        "grace_seconds": POWER_GRACE_S,
        "countdown": countdown,
    })


@app.route("/api/power/arm", methods=["POST"])
def api_power_arm():
    action = ((request.json or {}).get("action") or "none").lower()
    if action not in ("none", "sleep", "shutdown"):
        return jsonify({"error": "action must be none|sleep|shutdown"}), 400
    _write_power_action(action)
    power_state["idle_since"] = None
    power_state["saw_work"] = _any_work_running()  # if armed mid-run, latch right away
    power_state["firing"] = False
    power_state["fired_action"] = None
    if action != "none":
        add_log(f"Auto power-off armed: {action.upper()} when all work finishes.", "warn")
    else:
        add_log("Auto power-off disarmed.", "info")
    return jsonify({"status": "ok", "action": action, "work_running": power_state["saw_work"]})


@app.route("/api/power/cancel", methods=["POST"])
def api_power_cancel():
    """Disarm and abort any shutdown countdown already in progress."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen(["shutdown", "/a"], creationflags=flags)
    except Exception:
        pass
    _write_power_action("none")
    power_state.update(idle_since=None, saw_work=False, firing=False, fired_action=None)
    add_log("Auto power-off canceled / disarmed.", "info")
    return jsonify({"status": "canceled"})


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    """Kill switch: stop the dashboard server process. Any benchmark/training the
    dashboard launched keeps running (they're independent processes)."""
    add_log("Dashboard shutdown requested (kill switch)", "warn")

    def _die():
        time.sleep(0.4)  # let the HTTP response flush first
        os._exit(0)

    threading.Thread(target=_die, daemon=True).start()
    return jsonify({"status": "dashboard shutting down"})


@app.route("/api/shutdown/hard", methods=["POST"])
def api_shutdown_hard():
    """Hard kill: end the Scheduled Task so Windows won't auto-restart it,
    then exit the process. Use this when the normal Close button keeps reviving."""
    add_log("Hard kill requested — ending OCR2TeXDashboard scheduled task + exiting", "warn")

    def _hard_die():
        time.sleep(0.4)
        # End the scheduled task first (prevents auto-restart by Task Scheduler)
        try:
            subprocess.run(["schtasks", "/end", "/tn", "OCR2TeXDashboard"],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=_hard_die, daemon=True).start()
    return jsonify({"status": "hard kill initiated"})


if __name__ == "__main__":
    print("=" * 60)
    print("  OCR2TeX Dataset Pipeline Dashboard")
    print(f"  Source: {DATA_DIR}")
    print(f"  Workspace: {WORK_DIR}")
    print("=" * 60)
    source_files = get_image_files(DATA_DIR)
    print(f"  Found {len(source_files)} images in source directory")
    print(f"  Open http://127.0.0.1:5001 in your browser")
    print("=" * 60)
    _write_power_action("none")  # never auto-fire from a stale arm on a fresh start
    threading.Thread(target=_power_watcher, daemon=True).start()
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
