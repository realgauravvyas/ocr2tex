#!/usr/bin/env python3
"""
Baidu Unlimited-OCR Dashboard
==============================
Upload any image or PDF page and transcribe it using baidu/Unlimited-OCR (3B MIT model).
Supports single images (gundam/base mode) and multi-page PDFs.

Usage:
    python ocr_dashboard.py
    Open: http://localhost:7862

Requirements:
    pip install flask pillow pymupdf torch transformers einops addict easydict psutil
    (torch with CUDA 12.x recommended for GPU inference)
"""

import os
import sys
import uuid
import time
import shutil
import tempfile
import threading
import base64
import json
from io import BytesIO
from pathlib import Path

# Enable the Rust-based fast/robust downloader if available — fixes stalled
# HF downloads (many-connection hangs to S3). Must be set before transformers import.
try:
    import hf_transfer  # noqa
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
except ImportError:
    pass

try:
    from flask import Flask, send_file, jsonify, request
    from werkzeug.utils import secure_filename
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "flask", "werkzeug"], check=True)
    from flask import Flask, send_file, jsonify, request
    from werkzeug.utils import secure_filename

try:
    from PIL import Image as PILImage
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "Pillow"], check=True)
    from PIL import Image as PILImage

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE        = Path(__file__).parent
UPLOADS_DIR = BASE / 'uploads'
JOBS_DIR    = BASE / 'jobs'
STATIC_DIR  = BASE / 'static'

MODEL_NAME = 'baidu/Unlimited-OCR'

# ─── Model state ─────────────────────────────────────────────────────────────

_mdl = {
    'model':     None,
    'tokenizer': None,
    'status':    'not_loaded',   # not_loaded | loading | ready | error
    'load_time': None,
    'error':     None,
    'device':    None,
}
_mdl_lock = threading.Lock()

# ─── Download progress (TEMPORARY — for watching the first 6 GB download) ──────
# Measures bytes on disk in the HF cache vs the repo's total size. Remove this
# block (and the progress fields in /api/model/status) once the model is cached.

_dl = {'total_bytes': None}


def _model_cache_blobs() -> Path | None:
    """Path to this model's blobs/ dir in the HF cache (where data lands)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        name = 'models--' + MODEL_NAME.replace('/', '--')
        return Path(HF_HUB_CACHE) / name / 'blobs'
    except Exception:
        return None


def _downloaded_bytes() -> int:
    """Sum of all blob files (including .incomplete partials)."""
    blobs = _model_cache_blobs()
    if not blobs or not blobs.exists():
        return 0
    total = 0
    for f in blobs.iterdir():
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _total_bytes() -> int:
    """Total size of the repo's files (cached after first lookup)."""
    if _dl['total_bytes'] is not None:
        return _dl['total_bytes']
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(MODEL_NAME, files_metadata=True)
        _dl['total_bytes'] = sum((s.size or 0) for s in info.siblings)
    except Exception:
        _dl['total_bytes'] = 0
    return _dl['total_bytes']


def _do_load_model():
    import torch

    # transformers >= 4.46 removed is_torch_fx_available; patch it back so the
    # model's trust_remote_code can import it without crashing.
    try:
        from transformers.utils.import_utils import is_torch_fx_available  # noqa
    except ImportError:
        import transformers.utils.import_utils as _iutils
        import transformers.utils as _tu
        _iutils.is_torch_fx_available = lambda: False
        _tu.is_torch_fx_available     = lambda: False

    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval()
    if device == 'cuda':
        model = model.cuda()

    _mdl['model']     = model
    _mdl['tokenizer'] = tokenizer
    _mdl['device']    = device
    _mdl['load_time'] = round(time.time() - t0, 1)
    _mdl['error']     = None
    _mdl['status']    = 'ready'


def _read_output(output_path: Path, stem: str) -> str:
    """Read OCR text from output directory. Tries .md then .txt then any text file."""
    for ext in ('.md', '.txt'):
        p = output_path / f'{stem}{ext}'
        if p.exists():
            return p.read_text(encoding='utf-8', errors='replace').strip()
    # Fallback: first text file in output
    for p in sorted(output_path.iterdir()):
        if p.suffix in ('.md', '.txt') and p.stat().st_size > 0:
            return p.read_text(encoding='utf-8', errors='replace').strip()
    return ''


# ─── Job registry ─────────────────────────────────────────────────────────────

jobs: dict = {}
_jobs_lock = threading.Lock()


def _jset(job_id, **kw):
    with _jobs_lock:
        jobs[job_id].update(kw)


# ─── Inference worker ─────────────────────────────────────────────────────────

def _ocr_worker(job_id: str, image_paths: list[Path], job_dir: Path, mode: str):
    """
    mode: 'single' → model.infer() (gundam, best for one page)
          'multi'  → model.infer_multi() (for 2+ pages / PDFs)
    """
    t0 = time.time()
    try:
        # Wait for model if still loading
        deadline = time.time() + 300
        while _mdl['status'] == 'loading':
            time.sleep(1)
            if time.time() > deadline:
                _jset(job_id, status='error', message='Model load timed out (5 min).')
                return
        if _mdl['status'] != 'ready':
            _jset(job_id, status='error', message=f'Model not ready: {_mdl.get("error")}')
            return

        model     = _mdl['model']
        tokenizer = _mdl['tokenizer']
        out_dir   = job_dir / 'ocr_output'
        out_dir.mkdir(parents=True, exist_ok=True)

        _jset(job_id, status='running', message='Running Unlimited-OCR inference…')

        result_text = ''

        if mode == 'multi' and len(image_paths) > 1:
            # Multi-page inference
            ret = model.infer_multi(
                tokenizer,
                prompt='<image>Multi page parsing.',
                image_files=[str(p) for p in image_paths],
                output_path=str(out_dir),
                image_size=1024,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=1024,
                save_results=True,
            )
            if isinstance(ret, str) and ret.strip():
                result_text = ret.strip()
            elif isinstance(ret, dict):
                result_text = ret.get('text', '')
        else:
            # Single image inference (gundam mode — best for one page)
            img_path = image_paths[0]
            ret = model.infer(
                tokenizer,
                prompt='<image>document parsing.',
                image_file=str(img_path),
                output_path=str(out_dir),
                base_size=1024, image_size=640, crop_mode=True,
                max_length=32768,
                no_repeat_ngram_size=35, ngram_window=128,
                save_results=True,
            )
            if isinstance(ret, str) and ret.strip():
                result_text = ret.strip()
            elif isinstance(ret, dict):
                result_text = ret.get('text', '')

        # Fallback: read from output files if return value was None/empty
        if not result_text:
            result_text = _read_output(out_dir, image_paths[0].stem)

        elapsed = round(time.time() - t0, 1)
        _jset(job_id,
              status='done',
              message='Done!',
              text=result_text,
              elapsed=elapsed)

    except Exception as e:
        import traceback
        _jset(job_id, status='error', message=str(e), traceback=traceback.format_exc())


# ─── PDF → images helper ──────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    try:
        import fitz
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pymupdf"], check=True)
        import fitz

    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = out_dir / f'page_{i+1:04d}.png'
        page.get_pixmap(matrix=mat).save(str(out))
        paths.append(out)
    doc.close()
    return paths


# ─── App ──────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(STATIC_DIR))


@app.errorhandler(Exception)
def _json_error(e):
    import traceback
    code = getattr(e, 'code', 500)
    if not isinstance(code, int):
        code = 500
    return jsonify({'ok': False, 'error': str(e)}), 200


# ── Health ────────────────────────────────────────────────────────────────────

@app.route('/api/ping')
def api_ping():
    return jsonify({'ok': True, 'version': 'unlimited-ocr-dashboard-v1'})


# ── Model endpoints ───────────────────────────────────────────────────────────

@app.route('/api/model/status')
def api_model_status():
    resp = {
        'status':    _mdl['status'],
        'load_time': _mdl['load_time'],
        'device':    _mdl['device'],
        'error':     _mdl['error'],
    }
    # TEMPORARY: download progress while loading (and before weights are cached)
    if _mdl['status'] == 'loading':
        done  = _downloaded_bytes()
        total = _total_bytes()
        resp['download_gb'] = round(done / 1e9, 2)
        if total:
            resp['download_total_gb'] = round(total / 1e9, 2)
            resp['download_pct']      = min(99, int(done / total * 100))
    return jsonify(resp)


@app.route('/api/model/load', methods=['POST'])
def api_model_load():
    with _mdl_lock:
        if _mdl['status'] == 'ready':
            return jsonify({'ok': True, 'already': True, 'load_time': _mdl['load_time']})
        if _mdl['status'] == 'loading':
            return jsonify({'ok': False, 'message': 'Already loading'})
        _mdl['status'] = 'loading'

    def _bg():
        try:
            _do_load_model()
        except Exception as e:
            _mdl['status'] = 'error'
            _mdl['error']  = str(e)
            print(f'[Model load error] {e}', flush=True)

    threading.Thread(target=_bg, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Loading started'})


@app.route('/api/model/unload', methods=['POST'])
def api_model_unload():
    with _mdl_lock:
        if _mdl['model'] is not None:
            try:
                import gc, torch
                _mdl['model'].cpu()
                del _mdl['model']
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        _mdl.update(model=None, tokenizer=None, status='not_loaded',
                    load_time=None, error=None, device=None)
    return jsonify({'ok': True})


# ── Upload ────────────────────────────────────────────────────────────────────

ALLOWED_IMAGES = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
ALLOWED_ALL    = ALLOWED_IMAGES | {'.pdf'}


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f   = request.files['file']
    ext = Path(secure_filename(f.filename or 'file')).suffix.lower()
    if ext not in ALLOWED_ALL:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    uid     = uuid.uuid4().hex[:12]
    raw_dir = UPLOADS_DIR / uid
    raw_dir.mkdir(parents=True, exist_ok=True)

    if ext == '.pdf':
        pdf_path = raw_dir / 'input.pdf'
        f.save(str(pdf_path))
        pages_dir = raw_dir / 'pages'
        pages_dir.mkdir(exist_ok=True)
        page_paths = _pdf_to_images(pdf_path, pages_dir)
        pages = [
            {'index': i, 'url': f'/upload/{uid}/page/{i+1}'}
            for i in range(len(page_paths))
        ]
        return jsonify({'upload_id': uid, 'type': 'pdf', 'pages': pages, 'page_count': len(page_paths)})
    else:
        # Always normalise to input.png — run_ocr and the image route both
        # expect this exact name regardless of the uploaded extension.
        img_path = raw_dir / 'input.png'
        buf = BytesIO(f.read())
        with PILImage.open(buf).convert('RGB') as img:
            w, h = img.size
            img.save(str(img_path), format='PNG')
        return jsonify({
            'upload_id': uid, 'type': 'image',
            'url': f'/upload/{uid}/image',
            'width': w, 'height': h,
        })


# ── Serve uploaded files ──────────────────────────────────────────────────────

@app.route('/upload/<uid>/image')
def serve_upload_image(uid):
    for ext in ('input.png', 'input.jpg', 'input.jpeg'):
        p = UPLOADS_DIR / uid / ext
        if p.exists():
            return send_file(str(p))
    return 'Not found', 404


@app.route('/upload/<uid>/page/<int:page_num>')
def serve_upload_page(uid, page_num):
    p = UPLOADS_DIR / uid / 'pages' / f'page_{page_num:04d}.png'
    return send_file(str(p)) if p.exists() else ('Not found', 404)


# ── OCR job ───────────────────────────────────────────────────────────────────

@app.route('/api/run_ocr', methods=['POST'])
def api_run_ocr():
    data      = request.get_json(silent=True) or {}
    upload_id = data.get('upload_id', '')
    mode      = data.get('mode', 'single')        # 'single' | 'multi'
    pages     = data.get('pages', None)           # list of 1-based page numbers, or null = all

    if not upload_id:
        return jsonify({'error': 'No upload_id'}), 400

    upload_dir = UPLOADS_DIR / upload_id
    if not upload_dir.exists():
        return jsonify({'error': 'Upload not found'}), 404

    # Collect image paths
    pages_dir = upload_dir / 'pages'
    if pages_dir.exists():
        all_pages = sorted(pages_dir.glob('page_*.png'))
        if pages:
            selected = [all_pages[i-1] for i in pages if 1 <= i <= len(all_pages)]
        else:
            selected = all_pages
        image_paths = selected if selected else all_pages
    else:
        # Single image upload — accept input.png (normal) or any legacy input.*
        img = upload_dir / 'input.png'
        if not img.exists():
            legacy = sorted(upload_dir.glob('input.*'))
            if not legacy:
                return jsonify({'error': 'No image found'}), 404
            img = legacy[0]
        image_paths = [img]
        mode = 'single'

    if not image_paths:
        return jsonify({'error': 'No images to process'}), 400

    # Auto-upgrade to multi if >1 page
    if len(image_paths) > 1:
        mode = 'multi'

    job_id  = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    with _jobs_lock:
        jobs[job_id] = {
            'status':  'queued',
            'message': 'Queued…',
            'created': time.time(),
            'page_count': len(image_paths),
            'mode': mode,
        }

    # Auto-load model if needed
    with _mdl_lock:
        if _mdl['status'] == 'not_loaded':
            _mdl['status'] = 'loading'
            def _bg_load():
                try:
                    _do_load_model()
                except Exception as e:
                    _mdl['status'] = 'error'
                    _mdl['error']  = str(e)
            threading.Thread(target=_bg_load, daemon=True).start()
            _jset(job_id, status='loading_model', message='Auto-loading Unlimited-OCR model…')

    threading.Thread(
        target=_ocr_worker,
        args=(job_id, image_paths, job_dir, mode),
        daemon=True,
    ).start()

    return jsonify({'job_id': job_id})


@app.route('/api/job/<job_id>')
def api_job_status(job_id):
    with _jobs_lock:
        job = dict(jobs.get(job_id, {}))
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(job)


# ── Download result ───────────────────────────────────────────────────────────

@app.route('/api/download/<job_id>')
def api_download(job_id):
    with _jobs_lock:
        job = dict(jobs.get(job_id, {}))
    text = job.get('text', '')
    if not text:
        return 'No result', 404
    buf = BytesIO(text.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/markdown',
                     as_attachment=True, download_name=f'ocr_{job_id}.md')


# ── Static ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for d in [UPLOADS_DIR, JOBS_DIR, STATIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print('=' * 62)
    print('  Baidu Unlimited-OCR Dashboard')
    print('=' * 62)
    print(f'  Model      : {MODEL_NAME}')
    print(f'  Uploads    : {UPLOADS_DIR}')
    print(f'  Jobs       : {JOBS_DIR}')
    print()
    print('  Model will auto-load on first OCR request.')
    print()
    print('  ► Open  http://localhost:7862')
    print('=' * 62)

    app.run(host='0.0.0.0', port=7862, debug=False, threaded=True)
