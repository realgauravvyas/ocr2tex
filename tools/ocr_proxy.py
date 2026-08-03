"""Transparent image-resizing proxy for Ollama.

GLM-OCR v3.1/v4.1 were fine-tuned at a fixed budget of 1536 image tokens
(~1.2 megapixels). Full-resolution scans or phone photos are typically
8-14 MP -- far enough off-distribution that the model degenerates into
repeating the same line until it hits the output token cap. See the
model card for details: https://huggingface.co/ctogaurav/GLM_OCR-GGUF

Ollama's own CLI and API send images at whatever resolution the caller
gives them, with no resizing. This proxy sits between your client and
the real Ollama server, intercepts any base64-encoded images in Ollama's
/api/generate and /api/chat requests, resizes them to the model's
training budget, and forwards everything else unchanged.

Setup:
    pip install flask requests pillow

Run:
    python ocr_proxy.py                  # listens on :11500, forwards to :11434
    python ocr_proxy.py --port 11500 --ollama http://localhost:11434

Then point any Ollama client at the proxy instead of Ollama directly:

    # CLI — set once per shell session
    set OLLAMA_HOST=http://localhost:11500        (Windows cmd)
    $env:OLLAMA_HOST = "http://localhost:11500"    (PowerShell)
    export OLLAMA_HOST=http://localhost:11500      (bash)
    ollama run glm-ocr-v4.1 "...prompt... page.jpg"

    # Or any script/app calling Ollama's HTTP API: just use
    # http://localhost:11500 instead of http://localhost:11434

This only helps Ollama (CLI or API) and any app that lets you point it at
a custom Ollama host. It cannot intercept LM Studio's built-in chat window,
which talks directly to its internal engine with no configurable endpoint --
for that interface, resize images manually first with resize_for_ocr.py.
"""

import argparse
import base64
import io

from flask import Flask, Response, request
import requests
from PIL import Image

# Same budget as resize_for_ocr.py -- keep these in sync.
MAX_PIXELS = 1536 * (14 * 2) ** 2  # 1,204,224


def resize_b64_image(b64_data: str) -> str:
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    pixels = w * h

    if pixels <= MAX_PIXELS:
        return b64_data  # already small enough, don't re-encode losslessly for nothing

    scale = (MAX_PIXELS / pixels) ** 0.5
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    print(f"  [proxy] resized image {w}x{h} ({pixels/1e6:.1f} MP) "
          f"-> {new_size[0]}x{new_size[1]} ({new_size[0]*new_size[1]/1e6:.2f} MP)")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def resize_images_in_payload(payload: dict) -> dict:
    """Handles both Ollama API shapes: top-level 'images' (generate) and
    per-message 'images' (chat)."""
    if "images" in payload and isinstance(payload["images"], list):
        payload["images"] = [resize_b64_image(im) for im in payload["images"]]

    if "messages" in payload and isinstance(payload["messages"], list):
        for msg in payload["messages"]:
            if isinstance(msg, dict) and isinstance(msg.get("images"), list):
                msg["images"] = [resize_b64_image(im) for im in msg["images"]]

    return payload


def create_app(ollama_url: str) -> Flask:
    app = Flask(__name__)

    @app.route("/<path:path>", methods=["GET", "POST"])
    def proxy(path):
        target = f"{ollama_url}/{path}"

        if request.method == "POST" and request.is_json:
            payload = resize_images_in_payload(request.get_json())
            upstream = requests.post(target, json=payload, stream=True,
                                      params=request.args)
        else:
            upstream = requests.request(
                request.method, target, params=request.args,
                data=request.get_data(), headers={
                    k: v for k, v in request.headers if k.lower() != "host"
                }, stream=True,
            )

        return Response(
            upstream.iter_content(chunk_size=4096),
            status=upstream.status_code,
            content_type=upstream.headers.get("content-type"),
        )

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=11500,
                    help="port for this proxy to listen on (default: 11500)")
    ap.add_argument("--ollama", default="http://localhost:11434",
                    help="real Ollama server URL (default: http://localhost:11434)")
    args = ap.parse_args()

    app = create_app(args.ollama)
    print(f"OCR resize proxy listening on http://localhost:{args.port}")
    print(f"Forwarding to Ollama at {args.ollama}")
    print(f"Set OLLAMA_HOST=http://localhost:{args.port} before using the ollama CLI.")
    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
