"""GLM-OCR v3.1/v4.1 — public demo Space.

Wraps the verified-working llama-mtmd-cli inference path (not an unverified
Python binding) so this Space reproduces exactly the behavior confirmed
during development: https://github.com/realgauravvyas/ocr2tex

Design principle: every setting that caused a real, debugged failure during
development is fixed here, not left to the visitor to get wrong:
  - image resize to the 1536-image-token training budget (always on)
  - the exact fine-tuned prompt (always used, not user-editable)
  - no system prompt injected (this model was never fine-tuned with one)
  - greedy decoding by default (temp=0, repeat_penalty=1.0)
Generation parameters (temperature, repeat penalty, max tokens, context) are
exposed under Advanced settings for experimentation, defaulted to the values
this model was actually benchmarked with.
"""

import subprocess
import tempfile
import shutil
import subprocess as sp
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "ctogaurav/GLM_OCR-GGUF"
MODEL_DIR = Path("/data/models")
MAX_PIXELS = 1536 * (14 * 2) ** 2  # 1,204,224 -- training image-token budget

TRAINED_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into a complete, compilable LaTeX document. Ignore printed text, "
    "student info, page numbers, cancelled work and rough work. Output only LaTeX."
)

VERSIONS = {
    "v4.1 (lower CER, best content accuracy)": {
        "model": "v4.1/GLM-OCR-v4.1-Q8_0.gguf",
        "mmproj": "v4.1/mmproj-GLM-OCR-v4.1-Q8_0.gguf",
    },
    "v3.1 (higher PDF compile rate: 88.9%)": {
        "model": "v3.1/GLM-OCR-v3.1-Q8_0.gguf",
        "mmproj": "v3.1/mmproj-GLM-OCR-v3.1-Q8_0.gguf",
    },
}

_downloaded = {}


def get_model_paths(version: str):
    if version not in _downloaded:
        info = VERSIONS[version]
        model_path = hf_hub_download(REPO, info["model"], local_dir=str(MODEL_DIR))
        mmproj_path = hf_hub_download(REPO, info["mmproj"], local_dir=str(MODEL_DIR))
        _downloaded[version] = (model_path, mmproj_path)
    return _downloaded[version]


def resize_to_budget(img: Image.Image) -> tuple[Image.Image, str]:
    w, h = img.size
    pixels = w * h
    if pixels <= MAX_PIXELS:
        return img, f"{w}x{h} ({pixels/1e6:.2f} MP) — already within the {MAX_PIXELS/1e6:.1f} MP training budget"
    scale = (MAX_PIXELS / pixels) ** 0.5
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    note = (f"Resized {w}x{h} ({pixels/1e6:.1f} MP) -> {new_size[0]}x{new_size[1]} "
            f"({new_size[0]*new_size[1]/1e6:.2f} MP) to match the model's training budget. "
            f"Unresized images push the model off-distribution and cause degenerate, "
            f"repeating output — see the project README for why.")
    return resized, note


def try_compile(latex_src: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        tex_path = Path(tmp) / "out.tex"
        tex_path.write_text(latex_src, encoding="utf-8")
        try:
            result = sp.run(
                ["pdflatex", "-interaction=nonstopmode", "out.tex"],
                cwd=tmp, capture_output=True, text=True, timeout=30,
            )
        except sp.TimeoutExpired:
            return False, "pdflatex timed out"
        pdf_path = Path(tmp) / "out.pdf"
        if pdf_path.exists():
            return True, "Compiled successfully"
        # Surface the actual reason, not just "failed" -- honesty over a clean UI
        tail = result.stdout[-600:] if result.stdout else "(no output)"
        return False, f"Did not compile — pdflatex output (last 600 chars):\n{tail}"


def run_ocr(image: Image.Image, version: str, temperature: float,
            repeat_penalty: float, max_tokens: int, context_size: int):
    if image is None:
        return "Upload a page image first.", "", ""

    model_path, mmproj_path = get_model_paths(version)
    resized, resize_note = resize_to_budget(image)

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "page.png"
        resized.save(img_path)

        cmd = [
            "llama-mtmd-cli",
            "-m", model_path,
            "--mmproj", mmproj_path,
            "--image", str(img_path),
            "-p", TRAINED_PROMPT,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--repeat-penalty", str(repeat_penalty),
            "-c", str(context_size),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return resize_note, "(timed out after 300s — try a shorter page or fewer max tokens)", ""

        # llama-mtmd-cli writes its generation to stdout after its log lines;
        # the LaTeX body is what follows the last log-timestamp line.
        output = result.stdout.strip()

    compiled, compile_msg = try_compile(output)
    status = f"✅ {compile_msg}" if compiled else f"⚠️ {compile_msg}"
    return resize_note, output, status


with gr.Blocks(title="GLM-OCR — Handwritten Math to LaTeX") as demo:
    gr.Markdown(
        "# GLM-OCR — Handwritten Math → LaTeX\n"
        "Upload a photo or scan of handwritten math. Fine-tuned versions of "
        "[zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) — "
        "[model card](https://huggingface.co/ctogaurav/GLM_OCR) · "
        "[code](https://github.com/realgauravvyas/ocr2tex)\n\n"
        "**Honest limitation**: measured PDF compile rate on held-out data is "
        "82.4% (v4.1) / 88.9% (v3.1) — some pages will not produce valid LaTeX "
        "even with correct settings. This demo reports compile success/failure "
        "explicitly rather than hiding it."
    )

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Handwritten math page")
            version_in = gr.Dropdown(
                choices=list(VERSIONS.keys()),
                value=list(VERSIONS.keys())[0],
                label="Model version",
            )
            run_btn = gr.Button("Transcribe", variant="primary")

            with gr.Accordion("Advanced settings", open=False):
                gr.Markdown(
                    "Defaults match how this model was fine-tuned and "
                    "benchmarked. Changing them is for experimentation — "
                    "results are not guaranteed outside these defaults."
                )
                temperature_in = gr.Slider(0, 1.5, value=0, step=0.05, label="Temperature")
                repeat_penalty_in = gr.Slider(1.0, 1.5, value=1.0, step=0.05, label="Repeat penalty")
                max_tokens_in = gr.Slider(256, 4096, value=2048, step=256, label="Max output tokens")
                context_in = gr.Slider(2048, 16384, value=8192, step=1024, label="Context size")

        with gr.Column():
            resize_note_out = gr.Textbox(label="Image preprocessing", interactive=False)
            latex_out = gr.Code(label="Transcribed LaTeX", language="latex")
            status_out = gr.Textbox(label="Compile status", interactive=False)

    run_btn.click(
        run_ocr,
        inputs=[image_in, version_in, temperature_in, repeat_penalty_in,
                max_tokens_in, context_in],
        outputs=[resize_note_out, latex_out, status_out],
    )

if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)
