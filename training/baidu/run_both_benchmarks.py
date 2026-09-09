"""Run two benchmarks back-to-back, automatically, on the single GPU:
  1) Baidu OCR FT  — the fine-tuned model with the CURRENT (latest checkpoint) weights
  2) Baidu Unlimited-OCR — the base model (resumes 158 -> 700)
The unified scorer (running separately) scores both columns live as pages appear.
Runs as a Scheduled Task so it survives session boundaries.
"""
import subprocess, glob, os, time

BAIDU_PY = r"D:\Claude Code\BaiduOCR\venv\Scripts\python.exe"
DIR = r"D:\Claude Code\BaiduOCR"
ADAPTER = r"D:\Claude Code\BaiduOCR\finetune\baidu-ocr-math-v1\checkpoint"  # current trained weights
FT_OUT = r"D:\ocr2tex\output\bench_outputs\Baidu_OCR_FT"


def clear_ft():
    for pat in (FT_OUT + r"\*", r"D:\ocr2tex\output\bench_cache\Baidu_OCR_FT__*.jsonl"):
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except Exception:
                pass


# make sure the scorer is up (it populates both columns)
subprocess.run(["schtasks", "/run", "/tn", "OCR2TeXScorer"], capture_output=True)

# 1) Baidu OCR FT — fresh, with the current trained weights
print("=== [1/2] Baidu OCR FT benchmark (current weights @ checkpoint) ===", flush=True)
clear_ft()
subprocess.run([BAIDU_PY, "-u", DIR + r"\gen_baidu_ft_for_benchmark.py",
                "--samples", "700", "--adapter", ADAPTER, "--page-timeout", "120"])

# 2) Baidu Unlimited-OCR base — resumes 158 -> 700
print("=== [2/2] Baidu Unlimited-OCR base benchmark (resume to 700) ===", flush=True)
subprocess.run([BAIDU_PY, "-u", DIR + r"\gen_baidu_for_benchmark.py",
                "--samples", "700", "--page-timeout", "120"])

print("=== BOTH BENCHMARKS DONE ===", flush=True)
