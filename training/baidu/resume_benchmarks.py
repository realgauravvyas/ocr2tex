"""Resume both Baidu benchmarks WITHOUT clearing finished pages:
  1) Baidu OCR FT (best/final trained adapter) — resumes 523 -> 700
  2) Baidu Unlimited-OCR base — resumes 159 -> 700
Each generator skips pages already on disk, so this is safe to run repeatedly.
The unified scorer (separate task) fills both columns. Runs as a Scheduled Task.
"""
import subprocess

BAIDU_PY = r"D:\Claude Code\BaiduOCR\venv\Scripts\python.exe"
DIR = r"D:\Claude Code\BaiduOCR"
ADAPTER = r"D:\Claude Code\BaiduOCR\finetune\baidu-ocr-math-v1\best"  # same adapter the FT pages used

subprocess.run(["schtasks", "/run", "/tn", "OCR2TeXScorer"], capture_output=True)

print("=== [1/2] Baidu OCR FT (resume) ===", flush=True)
subprocess.run([BAIDU_PY, "-u", DIR + r"\gen_baidu_ft_for_benchmark.py",
                "--samples", "700", "--adapter", ADAPTER, "--page-timeout", "120"])

print("=== [2/2] Baidu Unlimited-OCR base (resume) ===", flush=True)
subprocess.run([BAIDU_PY, "-u", DIR + r"\gen_baidu_for_benchmark.py",
                "--samples", "700", "--page-timeout", "120"])

print("=== BOTH BENCHMARKS DONE ===", flush=True)
