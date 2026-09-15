# -*- coding: utf-8 -*-
"""
Generates a comprehensive Markdown report summarizing training and benchmark results.
"""

import sys
import json
import time
from pathlib import Path

def generate_report(output_dir="./output", gpu_name="NVIDIA A100", train_duration_s=0, bench_duration_s=0):
    out = Path(output_dir)
    bench_file = out / "benchmark_results_v5.json"
    metrics_file = out / "training_metrics.json"
    trainer_file = out / "trainer_state.json"
    report_file = out / "TRAINING_AND_BENCHMARK_REPORT.md"

    bench = {}
    if bench_file.exists():
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                bench = json.load(f)
        except Exception:
            pass

    metrics = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass

    trainer = {}
    if trainer_file.exists():
        try:
            with open(trainer_file, "r", encoding="utf-8") as f:
                trainer = json.load(f)
        except Exception:
            pass

    train_h = int(train_duration_s // 3600)
    train_m = int((train_duration_s % 3600) // 60)
    bench_m = int(bench_duration_s // 60)

    lines = [
        "# GLM-OCR v5.0 Training & Official Benchmark Verification Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"**Target Hardware:** {gpu_name}  ",
        f"**Total Training Runtime:** {train_h}h {train_m}m  ",
        f"**Total Benchmark Runtime:** {bench_m}m  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Verification",
        "",
        "| Metric Category | Metric | Result | Target / Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Character Accuracy** | **Mean CER** | **{bench.get('mean_cer', 'N/A')}** | < 0.10 |",
        f"| | **Median CER** | {bench.get('median_cer', 'N/A')} | Lowest error median |",
        f"| | **Normalized CER (NCER)** | **{bench.get('norm_cer', 'N/A')}** | Invariant to spacing |",
        f"| | **Character Similarity** | {round(bench.get('charsim', 0)*100, 1) if 'charsim' in bench else 'N/A'}% | High match fidelity |",
        f"| **Syntax & Document Structure** | **Structure Validity** | **{bench.get('struct_pct', 'N/A')}%** | > 95% Matching `{{}}` & `\\begin/\\end` |",
        f"| | **PDF Compile Rate** | {bench.get('compile_rate', 'N/A')}% | Valid pdflatex PDF renders |",
        f"| **Mathematical Semantics** | **Math-F1 Symbol Score** | **{bench.get('math_f1', 'N/A')}** | > 0.90 Equations & symbols |",
        f"| | **BLEU-4 Precision** | {bench.get('bleu', 'N/A')} | N-gram token overlap |",
        f"| | **ROUGE-L Token F1** | {bench.get('rouge_l', 'N/A')} | Longest sequence match |",
        f"| | **chrF n-gram F-score** | {bench.get('chrf', 'N/A')} | Character n-gram score |",
        f"| **Real-world Usability** | **CER < 10% (Clean Pass)** | {bench.get('cer_under_10pct', 'N/A')}% | Ready for production |",
        f"| | **CER < 30% (Usable)** | {bench.get('cer_under_30pct', 'N/A')}% | Requiring minor touches |",
        f"| **Inference Latency** | **Avg Page Latency** | **{bench.get('avg_latency_s', 'N/A')}s** | Per page on {gpu_name} |",
        "",
        "---",
        "",
        "## 2. Training Hyperparameters & Convergence",
        "",
        "- **Base Architecture:** `zai-org/GLM-OCR` (Vision-Language)",
        "- **Dual-Modality LoRA Config:** `r=32`, `alpha=64`, `dropout=0.05`",
        "- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `qkv`, `proj`",
        f"- **Completed Steps:** {trainer.get('global_step', 'N/A')} / {trainer.get('max_steps', 'N/A')}",
        f"- **Final Training Loss:** {metrics.get('last_loss', 'N/A')}",
        f"- **Final Validation Loss:** {metrics.get('last_val_loss', 'N/A')}",
        "- **Precision:** Native `bfloat16` with Ampere TF32 Tensor Cores",
        "- **Optimizer:** `adamw_torch_fused`",
        "",
        "---",
        "",
        "## 3. Package File Manifest",
        "",
        "Inside this archive (`final_results_package.tar.gz`), you have:",
        "1. `adapter/` - Full fine-tuned LoRA weights (`adapter_model.safetensors`, `adapter_config.json`)",
        "2. `benchmark_results_v5.json` - Complete machine-readable JSON metrics on all 700 test pages",
        "3. `training_metrics.json` - Complete loss and step logs across the training run",
        "4. `trainer_state.json` - PyTorch Trainer state and checkpoint history",
        "5. `TRAINING_AND_BENCHMARK_REPORT.md` - This executive summary",
        "6. `bench_outputs/` - Generated `.tex` source files for all 700 test pages",
    ]

    report_content = "\n".join(lines) + "\n"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"[REPORT] Executive verification report generated at: {report_file}")

if __name__ == "__main__":
    dur_train = float(sys.argv[1]) if len(sys.argv) > 1 else 0
    dur_bench = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    gpu = sys.argv[3] if len(sys.argv) > 3 else "NVIDIA A100"
    generate_report("./output", gpu, dur_train, dur_bench)
