# OCR2TeX — Handwritten Math → Compilable LaTeX

Fine-tunes [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B) with LoRA to
transcribe handwritten university-level math answer sheets into complete, `pdflatex`-compilable
LaTeX documents — ignoring printed headers, student identifiers, page numbers, and cancelled work.

Built as an internship project (B.Sc. Data Science and AI, IIT Guwahati). Full write-up: see
[Report](#report). Trained adapters: [ctogaurav/GLM_OCR](https://huggingface.co/ctogaurav/GLM_OCR)
on Hugging Face. Try it without installing anything: [Colab demos](#colab-demos).

## Headline results (700-page held-out benchmark)

| System | Mean CER ↓ | Compile % ↑ | Math-F1 ↑ | Latency (s) ↓ |
|---|---|---|---|---|
| Base GLM-OCR (frozen) | 0.515 | 0.0 | 0.703 | 8.38 |
| Baidu OCR (stock) | 0.718 | 42.7 | 0.626 | 40.79 |
| Baidu OCR (fine-tuned, same corpus) | 0.407 | 62.7 | 0.796 | 24.93 |
| **GLM-OCR v3.1 (ours)** | 0.397 | **88.9** | 0.817 | 14.12 |
| **GLM-OCR v4.1 (ours)** | **0.382** | 82.4 | **0.827** | 13.44 |

Fine-tuning cuts mean CER 25.9% relative and takes compile rate from **0% → 82.4%** — the base
model can read most of the math already, it just can't emit a document that renders. Full
metrics, methodology, and the honest v3.1-vs-v4.1 compile-rate tradeoff are in the
[report](#report) and the [HF model card](https://huggingface.co/ctogaurav/GLM_OCR).

CER is measured against silver labels from a teacher VLM, not human-verified ground truth —
treat it as teacher-agreement, not absolute accuracy (see report Discussion/Limitations).

## Repo layout

```
pipeline/     data curation — raw scans → validated, compilable training pairs
training/     LoRA fine-tuning: GLM-OCR (ours) and Baidu OCR (comparison baseline)
benchmark/    scoring harness — CER/BLEU/chrF/Math-F1/compile-rate on held-out pages
inspect/      QA tooling — handwriting classification, rejected-page triage, cleanup
dashboard/    Flask app: live browser UI for training + benchmarking (Steps 4 below)
colab/        Google Colab demo notebooks (upload a page, compare base vs fine-tuned)
samples/      2 PII-verified sample pages, traced through the pipeline (see samples/README.md)
```

## 1. Data pipeline (`pipeline/`)

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

⚠️ **Known limitation**: `anonymize.py` redacts PII by whiting out a fixed top-% of each page —
positional, not content-aware. It reliably catches printed headers on standard answer pages but
can miss non-standard layouts (e.g. a signature field on a cover/instructions page). Every image
in `samples/` was manually verified beyond this automated step; the full dataset relies on the
automated step alone and is **not published** in this repo for that reason.

Labels come from a commercial teacher VLM accessed through an auto-routing API
(`orcarouter/auto`) — set `ORCAROUTER_API_KEY` (see `.env.example`).

## 2. Fine-tuning

Two model families are trained and compared, both via LoRA (r=32, α=64, dropout=0.05,
targets: q/k/v/o/gate/up/down projections) on a single RTX 3060 12GB.

### GLM-OCR (`training/glm_ocr/`) — the model this project ships

| | v3.1 | v4.1 |
|---|---|---|
| Training pages | 4,672 | 12,575 |
| Learning rate | 2e-5 | 1e-5 |
| Warm start | from the v3 adapter | from the v4 adapter |
| Steps | 1,168 | 3,144 |
| Best val loss | 0.108 | 0.164 |

```
python training/glm_ocr/train_glm_ocr.py
```

### Baidu OCR (`training/baidu/`) — comparison baseline, same training corpus

Fine-tuned on the **identical 12,575-page corpus** as GLM-OCR v4.1, so the comparison in the
results table above isolates model/recipe differences, not data access. `train_unlimited_ocr.py`
is the original v1 run; `train_baidu_v2.py` + `gen_baidu_ft_v2*.py` produced the v2 variant.

## 3. Benchmark (`benchmark/`)

Scores every system on the 700-page held-out test split. 15 metrics logged in
`benchmark_results.json` (source of truth for every number in this README, the report, and the
HF card): mean/median CER, style-normalised CER, character similarity, token error, CER<10%/30%,
BLEU-4, ROUGE-L, chrF, Math-F1, structural-match %, compile rate, length ratio, latency.

```
python benchmark/benchmark_glm_ocr.py     # score a GLM-OCR checkpoint
python benchmark/score_baidu_live.py      # score a Baidu OCR checkpoint
python benchmark/score_models_live.py     # aggregate + compare all systems
```

## 4. Inspect / QA (`inspect/`)

Tooling used during data curation to catch and fix problems the automated pipeline missed —
this is what makes Stage 7 ("Automated quality review") in the pipeline table above real:

| Script | Purpose |
|---|---|
| `classify_handwriting.py` | Flags pages that are entirely printed (should've been filtered at Stage 2 but weren't) |
| `analyze_rejected.py` | Inspects why pages failed `pdflatex` validation |
| `check_v41_printed.py` | Re-checks the v4.1 split for the same printed-page leakage |
| `clean_annotations.py` | Strips malformed/truncated teacher-VLM output |
| `requeue_rejected.py` | Sends rejected pages back through re-annotation |
| `verify_salvaged.py` | Confirms salvaged pages actually compile before re-inclusion |
| `build_salvaged_dataset.py`, `build_v41_dataset.py` | Rebuild JSONL splits after a salvage pass |

## 5. Dashboard (`dashboard/`)

Live browser UI combining fine-tuning control, benchmark scoring, and dataset inspection in one
Flask app — this is the actual tool used to run and monitor every experiment in this project.

```
pip install -r requirements.txt
python dashboard/app.py
# open http://127.0.0.1:5001
```

## Colab demos

Upload a page of your own handwriting and compare base vs. fine-tuned output, no local setup:

- [Colab v3.1](https://colab.research.google.com/drive/1-uW5d5C9cRrGnvLMpE7cko0wnrygBQma)
- [Colab v4.1](https://colab.research.google.com/drive/1SC0mfy98CQdm3ARDd3zuD8EGrWKgnl5-)

Local copies (with real adapter Drive IDs already filled in) are in `colab/notebooks/`.

## Samples

`samples/` has two pages traced through the pipeline (raw crop → prepared → annotated →
validated) — see `samples/README.md` for the PII-safety note on how these were selected. The full
34,080-scan dataset is confidential and is **not** included in this repo.

## Report

Full IEEE-format writeup — methodology, all results tables, discussion of the v3.1/v4.1
compile-rate tradeoff, and limitations — is maintained separately (Overleaf-managed) and not
duplicated here.

## Model weights

Trained adapters (v3.1 and v4.1) are hosted on Hugging Face, not in this repo:
[huggingface.co/ctogaurav/GLM_OCR](https://huggingface.co/ctogaurav/GLM_OCR) — MIT licensed,
usable commercially.

**Want to run v4.1 in LM Studio / Ollama / llama.cpp?** A ready-to-download, verified-working
GGUF is at [ctogaurav/GLM_OCR-GGUF](https://huggingface.co/ctogaurav/GLM_OCR-GGUF) — tested
end-to-end on GPU, output checked against this repo's own `samples/` ground truth, not just
"it loaded." Producing it required patching a real bug in llama.cpp's own conversion code for
this architecture; that fix is documented there too.

## Environment

Everything — pipeline, training, and benchmarking — ran on a single personal machine, no cloud
compute.

| | |
|---|---|
| GPU | NVIDIA RTX 3060, 12GB VRAM |
| Python | 3.11.9 |
| PyTorch | 2.10.0+cu130 |
| Transformers | 5.9.0 |
| PEFT | 0.18.1 (verified against both v3.1's and v4.1's `adapter_config.json`) |
| LaTeX | MiKTeX (`pdflatex`, used for Stage 8 validation and benchmark compile-rate scoring) |

**Training time**: v3.1 — ~5 hours (1,168 steps). v4.1 — ~12.7 hours (3,144 steps; estimated from
steps × sec/step, since the run was interrupted and resumed across multiple sessions on shared
personal hardware — the logged "elapsed" field only covers the final resumed segment).

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # fill in your own API keys — never commit .env
```

## AI-assistance note

AI-assisted tools were used for code scaffolding and drafting support during this project.
Pipeline design, experimental protocol, analysis, and conclusions are the author's own; all
reported numbers were produced by the author's own runs of the benchmark harness in this repo.
