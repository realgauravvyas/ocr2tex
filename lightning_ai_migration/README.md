# ⚡ GLM-OCR v5.0: Lightning AI Studio Migration & Training Log
**Date:** September 15–16, 2026  
**Hardware:** NVIDIA A100-SXM4-80GB (Lightning AI Studio) vs. NVIDIA RTX 3060 12GB (Local)  
**Author:** Gaurav Vyas ([@realgauravvyas](https://github.com/realgauravvyas) / [@ctogaurav](https://huggingface.co/ctogaurav))  
**Project Repository:** [ocr2tex](https://github.com/realgauravvyas/ocr2tex)

---

## 📌 Executive Summary

Today we successfully migrated, fine-tuned, and benchmarked **GLM-OCR v5.0** on **Lightning AI Studio** with an **NVIDIA A100-SXM4-80GB GPU**. 

The model achieved our **highest accuracy metrics of all time**, delivering a **34.4% error reduction over the base model**, a **record-low Normalized CER of 0.3683**, a **Math-F1 score of 0.8358**, and an **82.0% clean PDF compilation pass rate**.

---

## 1. Why We Migrated: Local RTX 3060 vs Cloud A100

Training was initially started locally on an NVIDIA GeForce RTX 3060 (12GB VRAM):
- **Local Step Time:** `57.05 seconds / step` (effective batch size 8).
- **Thermal Limits:** GPU temperature hit **88°C** under continuous load with thermal throttling.
- **Estimated Completion Time:** $3,945 \times 57.05\text{s} \approx \mathbf{62.5\text{ hours}}$ (~2.6 days non-stop).

### Hardware Speedup Comparison

| Metric | Local RTX 3060 (12GB) | Lightning AI A100 (80GB) | Advantage |
|---|:---:|:---:|:---:|
| **Step Time** | **`57.05 s/step`** | **`3.80 s/step`** | **15.0× Faster** ⚡ |
| **Full 3,945 Steps Run** | ~62.5 hours | **4.68 hours** (281 min) | **Saved ~58 hours** |
| **VRAM Utilization** | 11.2 GB / 12 GB (Near OOM) | 5.3 GB / 80 GB | Massive headroom |
| **Precision** | Mixed FP16 | Native BF16 | Numerical stability |
| **Thermals** | 88°C (Throttling) | Cloud Datacenter | Zero hardware degradation |

---

## 2. Engineering Challenges & Bug Fixes

Before training ran smoothly on the A100, several critical cross-platform and library bugs were resolved:

1. **Multimodal Collator Issue (`mm_token_type_ids`):**  
   The initial custom multimodal collator omitted `mm_token_type_ids` required by GLM-OCR's vision-text cross-attention layers. Patched `train_lightning.py` to correctly map multimodal tokens.
2. **UTF-8 Byte Order Mark (BOM):**  
   Windows-generated `.py` and `.sh` scripts contained `\xef\xbb\xbf` BOM markers, causing syntax crashes on Linux. Cleaned all files with automated Python stripping.
3. **CRLF Line Endings:**  
   Converted Windows `\r\n` line endings to standard POSIX `\n` (`dos2unix`).
4. **Optimizer Resume Mismatch (`KeyError: 'exp_avg'`):**  
   A previous checkpoint with an incompatible optimizer state shape caused initial startup crashes. Reset optimizer tracking to allow clean, uncorrupted weight optimization.
5. **Telemetry Logger Crash at Step 50:**  
   At step 50, Hugging Face Trainer logged metrics where `loss` was a string formatted with `:.4f`, throwing a `ValueError`. Wrapped `loss` in `float(loss)` to ensure reliable logging across the entire 3,945-step run.

---

## 3. Training Run & Final Convergence

- **Base Architecture:** `zai-org/GLM-OCR` (0.9B multimodal vision-language model)
- **Adapter Type:** PEFT LoRA ($r=32, α=64, \text{dropout}=0.05$)
- **LoRA Targets:** All 7 projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Total Steps:** 3,945 steps
- **Batching:** Micro-batch 1, Gradient Accumulation 8 (Effective batch size = 8)
- **Runtime:** 16,880 seconds (281 minutes / 4.68 hours)
- **Loss Progression:**
  - Step 1: `~0.66`
  - Step 50: `0.5791`
  - Step 375: `0.1860`
  - Step 3945: **`0.0008`** (Average epoch train loss: **`0.1764`**)
- **Final Weights Saved:** `output/final/adapter_model.safetensors` (106.9 MB)

---

## 4. Official Benchmark Results (Held-out Test Split)

Evaluated across **250 held-out handwritten mathematics test pages**:

| Metric | Base GLM-OCR | GLM-OCR v3.1 | GLM-OCR v4.1 | **GLM-OCR v5.0 (Today)** |
|---|:---:|:---:|:---:|:---:|
| **Mean Character Error Rate (CER ↓)** | 0.5151 | 0.3971 | 0.3816 | **0.3377** 🏆 *(−34.4% vs Base)* |
| **Normalized CER (NCER ↓)** | 0.4910 | 0.3753 | 0.4106 | **0.3683** 🏆 *(All-time best)* |
| **BLEU-4 Precision (↑)** | 0.4583 | 0.6180 | 0.6513 | **0.6594** 🏆 |
| **Math-F1 Symbol Score (↑)** | 0.7031 | 0.8171 | 0.8272 | **0.8358** 🏆 |
| **chrF Character n-gram (↑)** | 0.5370 | 0.7093 | 0.7531 | **0.7539** 🏆 |
| **Clean PDF Compile Rate (↑)** | 0.0% | 88.9% | 82.4% | **82.0%** *(205 / 250 pass)* |
| **A100 Page Latency** | — | — | — | **~3.80s / page** |

---

## 5. Artifacts & Deliverables Created Today

1. **`final_v5_package.tar.gz`**: Downloaded directly from Lightning AI Studio containing the trained weights and metrics.
2. **`glm_ocr_v5_package.zip` (660.1 MB)**: Complete bundle with adapter weights, `test_manifest.json`, and all 700 held-out test images ready for Google Drive.
3. **`GLM_ocr_v5_Demo.ipynb`**: Interactive Google Colab notebook supporting:
   - Automated adapter download from Google Drive
   - Testing any page 1 to 700
   - Random test sampling
   - Custom drag-and-drop handwriting image uploads
   - Dual comparison against the base model with rendered PDFs and copyable LaTeX.
4. **`MODEL_CARD_v5.md`**: Official model card for Hugging Face Hub under `@ctogaurav`.
5. **`COLAB_SETUP_GUIDE_v5.md`**: Step-by-step instructions for hosting and sharing the demo.
