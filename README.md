# OCR2TeX — Handwritten Math → Compilable LaTeX

Fine-tunes [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B vision-language model) with LoRA to
transcribe handwritten university-level math answer sheets into complete, `pdflatex`-compilable
LaTeX documents — ignoring printed headers, student identifiers, page numbers, and cancelled work.

Built as an undergraduate research/internship project (B.Sc. Data Science and AI, IIT Guwahati).
- **Author:** Gaurav Vyas ([@realgauravvyas](https://github.com/realgauravvyas) / [@ctogaurav](https://huggingface.co/ctogaurav))
- **Trained Adapters (v3.1, v4.1, v5.0):** [huggingface.co/ctogaurav/GLM_OCR](https://huggingface.co/ctogaurav/GLM_OCR)
- **Quantized GGUFs (LM Studio / Ollama):** [huggingface.co/ctogaurav/GLM_OCR-GGUF](https://huggingface.co/ctogaurav/GLM_OCR-GGUF)
- **Try it live (Zero local setup):** [Colab demos](#colab-demos) (supports testing all 700 test pages)

---

## 🏆 Headline Results (Held-Out Benchmark Comparison)

Evaluated on the held-out test split (`test.jsonl`). Metrics include Character Error Rate (CER), Normalized CER (NCER), Math symbol F1, and clean PDF compilation rate:

| System / Model | Mean CER ↓ | Norm CER ↓ | Compile % ↑ | Math-F1 ↑ | BLEU-4 ↑ | Latency (s) ↓ | Hardware |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base GLM-OCR (frozen) | 0.5151 | 0.4910 | 0.0 | 0.7031 | 0.4583 | 8.38 | RTX 3060 |
| Baidu OCR (stock) | 0.7176 | 0.7343 | 42.7 | 0.6264 | 0.3113 | 40.79 | API |
| Baidu OCR (fine-tuned v2) | 0.4258 | 0.4706 | 64.9 | 0.7983 | 0.5958 | 29.00 | RTX 3060 |
| **GLM-OCR v3.1 (ours)** | 0.3971 | 0.3753 | **88.9** | 0.8171 | 0.6180 | 14.12 | RTX 3060 |
| **GLM-OCR v4.1 (ours)** | 0.3816 | 0.4106 | 82.4 | 0.8272 | 0.6513 | 13.44 | RTX 3060 |
| **GLM-OCR v5.0 (ours, SOTA)** | **0.3377** 🏆 | **0.3683** 🏆 | **82.0** | **0.8358** 🏆 | **0.6594** 🏆 | **~3.80** | **A100 (80GB)** |

> **Key Takeaways for v5.0:**
> - **34.4% Relative CER Reduction** over the un-finetuned base model (0.5151 → 0.3377) and **11.5% reduction** over v4.1.
> - **Lowest Normalized CER (0.3683)**: Eliminates stylistic spacing differences, proving superior core LaTeX transcription.
> - **82.0% Clean Compile Rate**: 205 out of 250 tested documents compiled into pristine PDFs without manual syntax fixing.
> - **High Mathematical Precision**: Record high **0.8358 Math-F1** and **0.6594 BLEU-4**.

---

## 📁 Repo Layout

```
colab/                  Interactive Google Colab notebooks (v3.1, v4.1, and v5.0)
lightning_ai_migration/ Cloud A100 training scripts, telemetry fixes & migration logs
pipeline/               Data curation — raw scans → validated, compilable training pairs
training/               LoRA fine-tuning: GLM-OCR and Baidu OCR training scripts
benchmark/              Scoring harness — CER/BLEU/chrF/Math-F1/compile-rate evaluation
inspect/                QA tooling — handwriting classification, rejected-page triage
dashboard/              Flask app: live browser UI for local training + benchmarking
samples/                PII-verified sample pages traced through the pipeline
```

---

## 1. Data Pipeline (`pipeline/`)

34,080 raw scans → 13,973 validated image–LaTeX pairs. Every training target is verified to
compile with `pdflatex` before it's used — the model never trains on a broken target.

| # | Stage | Script | Kept | % |
|---|---|---|---|---|
| 1 | Crop / redact PII | `anonymize.py` | 34,080 | 100.0 |
| 2 | Filter blank / printed-only | `filter_by_filesize.py`, `sparse_page.py`, `whiteout_blank_and_nonblank_sorting.py` | 15,829 | 46.4 |
| 3–5 | Deskew → PNG → resize, rename to `page_NNNNN` | `deskew_smart.py`, `jpg2png.py`, `01_prepare_images.py` | 15,829 | 46.4 |
| 6 | Teacher-VLM annotation | `02_annotate_helper.py` | 15,461 | 45.4 |
| 7 | Automated quality review | `03_build_dataset.py` | 14,772 | 43.3 |
| 8 | `pdflatex` validation | `04_validate_dataset.py` | 14,560 | 42.7 |
| 9 | Train/val/test split | `05_split_dataset.py` | 13,973 | 41.0 |

**train / val / test = 12,575 / 698 / 700**

⚠️ **PII Redaction**: `anonymize.py` redacts PII by whiting out a fixed top-% of each page. The full 34,080-scan dataset is confidential and is **not published** in this repo for student privacy.

---

## 2. Fine-Tuning & Model Training

We train and compare three major iterations of GLM-OCR using Low-Rank Adaptation (LoRA):

### GLM-OCR Training Specifications

| Hyperparameter / Detail | v3.1 | v4.1 | **v5.0 (Latest SOTA)** |
|---|:---:|:---:|:---:|
| **Training Pages** | 4,672 | 12,575 | **12,575+** |
| **Compute Hardware** | Local RTX 3060 (12GB) | Local RTX 3060 (12GB) | **Hybrid: Local RTX 3060 (Phase 1) ➔ Cloud A100 (Phase 2)** |
| **Warm Start Strategy** | From v3 adapter | From v4 adapter | **Warm start from step 250 (local RTX 3060 checkpoint)** |
| **Learning Rate** | 2e-5 | 1e-5 | **1e-5 (Cosine decay with linear warmup)** |
| **Precision** | FP16 mixed | FP16 mixed | **FP16 (Local) ➔ Native BF16 (A100)** |
| **Effective Batch Size** | 8 (Batch 1 × Accum 8) | 8 (Batch 1 × Accum 8) | **8 (Batch 1 × Accum 8)** |
| **LoRA Rank ($r$) / Alpha ($lpha$)** | r=32, α=64 | r=32, α=64 | **r=32, α=64, dropout=0.05** |
| **Target Projections** | All 7 linear layers | All 7 linear layers | **q, k, v, o, gate, up, down projections** |
| **Total Training Steps** | 1,168 | 3,144 | **3,945 steps** |
| **Final Loss** | 0.108 (val) | 0.164 (val) | **0.0008 (step loss) / 0.1764 (avg train loss)** |
| **Step Speed** | ~45–50 s / step | ~57 s / step | **57.05 s/step (Local) ➔ 3.80 s/step (A100)** ⚡ |
| **Total Training Time** | ~5 hours | ~12.7 hours | **~4.68 hours on A100 (saved ~58 hours)** |

### 🖥️ Local Workstation Setup vs. Cloud Handoff (v5.0):

Training for v5.0 began locally on an **NVIDIA GeForce RTX 3060 12GB**:
- **Local Micro-Settings:** Micro-batch size `1`, Gradient Accumulation `8` (effective batch `8`), FP16 mixed precision, `max_length=3584`, `max_image_tokens=1536`.
- **Local Thermal Profile:** VRAM was nearly saturated at **11.2 GB / 12 GB**, and GPU core temperature hit **88°C** under continuous load, inducing thermal throttling (~57.05s/step).
- **Warm Startup (Step 250 Handoff):** The first **250 steps** were trained on the local RTX 3060 (checkpoint saved at loss ~0.42). To protect local hardware from a 62-hour continuous thermal ordeal, training was transitioned to an **NVIDIA A100-SXM4-80GB** on Lightning AI Studio, warm-starting from the 250-step state and accelerating the remaining steps at **3.80s/step** down to a final convergence loss of **0.0008**.

---|:---:|:---:|:---:|
| **Training Pages** | 4,672 | 12,575 | **12,575+** |
| **Compute Hardware** | Local RTX 3060 (12GB) | Local RTX 3060 (12GB) | **NVIDIA A100-SXM4-80GB (Lightning AI)** |
| **Learning Rate** | 2e-5 | 1e-5 | **1e-5 (Cosine decay with warmup)** |
| **Precision** | FP16 mixed | FP16 mixed | **Native BF16** |
| **Effective Batch Size** | 8 (Batch 1 × Accum 8) | 8 (Batch 1 × Accum 8) | **8 (Batch 1 × Accum 8)** |
| **LoRA Rank ($r$) / Alpha ($lpha$)** | r=32, α=64 | r=32, α=64 | **r=32, α=64, dropout=0.05** |
| **Target Projections** | All 7 linear layers | All 7 linear layers | **q, k, v, o, gate, up, down projections** |
| **Total Training Steps** | 1,168 | 3,144 | **3,945 steps** |
| **Final Loss** | 0.108 (val) | 0.164 (val) | **0.0008 (step loss) / 0.1764 (avg train loss)** |
| **Step Speed** | ~45–50 s / step | ~57 s / step | **3.80 s / step (15.0× faster!)** ⚡ |
| **Total Training Time** | ~5 hours | ~12.7 hours | **281 minutes (~4.68 hours)** |

> 💡 **Cloud Scaling Impact (v5.0):**  
> Running 3,945 steps on the local RTX 3060 would have required **~62.5 hours (~2.6 full days)** at 88°C thermal limit. Migrating to the cloud A100 reduced step latency from **57.05s → 3.80s**, finishing the entire run in **under 4.7 hours** and saving ~58 hours of compute time. Full migration scripts, collator patches, and logs are documented in [`lightning_ai_migration/README.md`](./lightning_ai_migration/README.md).

---

## 3. Benchmarking & Scoring (`benchmark/`)

Scored against the **700 held-out test split** (`test.jsonl`).
- For **v5.0**, evaluation was executed across **250 representative held-out test pages**:
  - **205 out of 250 pages compiled cleanly** into PDFs (**82.0% compile rate**).
  - **Mean CER:** `0.3377` | **Median CER:** `0.2858`
  - **Normalized CER (NCER):** `0.3683`
  - **BLEU-4 Precision:** `0.6594`
  - **Math-F1 Symbol Score:** `0.8358`
  - **chrF Score:** `0.7539`
  - **CER < 10% (near-perfect transcription):** `6.0%` of pages
  - **CER < 30% (immediately usable):** `52.4%` of pages

---

## 4. Colab Demos & Interactive Studio

Try the models live in Google Colab on a free GPU without installing anything locally:

- 🚀 **[GLM-OCR v5.0 Interactive Studio (Colab)](./colab/notebooks/GLM_ocr_v5_Demo.ipynb)**:
  - **Test any page (1 to 700):** Select any held-out page index to view the handwritten note, ground-truth reference, and compiled PDF side-by-side.
  - **Random Page Mode:** Draw random samples from the 700-page test split.
  - **Custom Image Upload:** Upload your own handwritten math pages/scans to transcribe and compile.
  - **Dual-Model Comparison:** Compare Base GLM-OCR (0.9B) vs. Fine-Tuned v5.0 in real-time.
- **[Colab v4.1 Demo](https://colab.research.google.com/drive/1SC0mfy98CQdm3ARDd3zuD8EGrWKgnl5-)**
- **[Colab v3.1 Demo](https://colab.research.google.com/drive/1-uW5d5C9cRrGnvLMpE7cko0wnrygBQma)**

---

## 5. Model Weights & Downloads

### Hugging Face LoRA Adapters
Official fine-tuned adapters are hosted at **[huggingface.co/ctogaurav/GLM_OCR](https://huggingface.co/ctogaurav/GLM_OCR)** (MIT License):
- **`v5.0/`**: SOTA adapter (`adapter_model.safetensors`, 106.9 MB)
- **`v4.1/`**: Intermediate adapter
- **`v3.1/`**: 4,672-page adapter

### GGUF Quantized Models (for LM Studio / Ollama / llama.cpp)
Ready-to-run GGUF quants are hosted at **[huggingface.co/ctogaurav/GLM_OCR-GGUF](https://huggingface.co/ctogaurav/GLM_OCR-GGUF)**:
- **`v5.0/GLM-OCR-v5.0-Q8_0.gguf`** (~682 MB) + **`v5.0/mmproj-GLM-OCR-v5.0-Q8_0.gguf`** (~484 MB)
- **`v5.0/Modelfile`**: Ready for `ollama create glm-ocr-v5.0 -f Modelfile`.
- Also includes `v4.1/` and `v3.1/` GGUF builds.

---

## 6. Environment & Hardware

| Spec | Local Workstation (v3.1, v4.1) | Cloud Cluster (v5.0 SOTA) |
|---|---|---|
| **GPU** | NVIDIA GeForce RTX 3060 (12GB VRAM) | NVIDIA A100-SXM4 (80GB VRAM) |
| **Platform** | Windows 11 / WSL2 | Ubuntu 22.04 LTS (Lightning AI Studio) |
| **Python** | 3.11.9 | 3.10.12 |
| **PyTorch** | 2.10.0+cu130 | 2.5.1+cu124 |
| **Transformers** | 5.9.0 | 4.49.0 |
| **PEFT** | 0.18.1 | 0.14.0 |
| **LaTeX Engine** | MiKTeX (`pdflatex`) | TeX Live 2023 (`pdflatex`) |

---

## Setup & Local Usage

```bash
git clone https://github.com/realgauravvyas/ocr2tex.git
cd ocr2tex
pip install -r requirements.txt
cp .env.example .env
```

---

## License & Attribution

- Released under the **MIT License**.
- Base vision-language model: [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR).
- Author: **Gaurav Vyas** ([GitHub](https://github.com/realgauravvyas) | [Hugging Face](https://huggingface.co/ctogaurav)).
