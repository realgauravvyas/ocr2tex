import os
import base64
import json
import random
from pathlib import Path
from openai import OpenAI

API_KEY = os.environ.get("TOKENROUTER_API_KEY", "")
API_BASE = "https://api.tokenrouter.com/v1"
MODEL = "MiniMax-M3"

source = Path(r"D:\ocr2tex\Data\raw data")
images = sorted([f for f in source.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
test_image = random.choice(images)

print(f"Testing API with: {test_image.name}")
print(f"  File size: {test_image.stat().st_size / 1024:.1f} KB")
print(f"  API: {API_BASE}")
print(f"  Model: {MODEL}")
print()

with open(test_image, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

print(f"  Image encoded: {len(img_b64)} chars base64")
print(f"  Sending request...")

SYSTEM_PROMPT = """You are an expert OCR system specialized in reading handwritten mathematics from undergraduate-level answer sheets.

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

USER_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into LaTeX. Ignore all printed text, student info, page numbers, "
    "cancelled work, and rough work. Preserve the spatial layout exactly. "
    "Output a complete compilable LaTeX document only."
)

try:
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    
    import time
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.1,
    )
    elapsed = time.time() - start
    
    raw = response.choices[0].message.content
    print(f"\n  RESPONSE TIME: {elapsed:.1f}s")
    print(f"  Response length: {len(raw)} chars")
    print(f"  Model used: {response.model}")
    
    if hasattr(response, 'usage') and response.usage:
        print(f"  Tokens - prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}, total: {response.usage.total_tokens}")
    
    text = raw.strip()
    if text.startswith("```latex"):
        text = text[len("```latex"):].strip()
    elif text.startswith("```tex"):
        text = text[len("```tex"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    
    has_docclass = "\\documentclass" in text
    has_enddoc = "\\end{document}" in text
    has_math = "$" in text or "\\[" in text or "\\begin{" in text
    
    print(f"\n  === VALIDATION ===")
    print(f"  Has \\documentclass: {'YES' if has_docclass else 'NO'}")
    print(f"  Has \\end{{document}}: {'YES' if has_enddoc else 'NO'}")
    print(f"  Has math content: {'YES' if has_math else 'NO'}")
    print(f"  Clean length: {len(text)} chars")
    
    if has_docclass and has_enddoc and has_math and len(text) > 200:
        print(f"\n  RESULT: API IS WORKING - valid LaTeX response")
    else:
        print(f"\n  RESULT: API RETURNED BUT RESPONSE MAY BE INVALID")
    
    print(f"\n  === FIRST 500 CHARS ===")
    print(f"  {text[:500]}")
    
except Exception as e:
    print(f"\n  ERROR: {e}")
    print(f"\n  RESULT: API IS NOT WORKING")
