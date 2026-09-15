---
license: mit
base_model: zai-org/GLM-OCR
language:
- en
pipeline_tag: image-to-text
tags:
- ocr
- latex
- mathematics
- handwritten-ocr
- lora
- peft
- vision-language-models
---

# GLM-OCR v5.0: Fine-Tuned Handwritten Math OCR → LaTeX (SOTA)

**Author:** Gaurav Vyas ([@realgauravvyas](https://huggingface.co/realgauravvyas))  
**GitHub Repository:** [realgauravvyas/ocr2tex](https://github.com/realgauravvyas/ocr2tex)  
**Interactive Colab Demo:** [GLM_ocr_v5_Demo.ipynb](https://github.com/realgauravvyas/ocr2tex/tree/main/colab/notebooks)  

---

## 📌 Overview

**GLM-OCR v5.0** is a state-of-the-art multimodal LoRA fine-tune of [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B vision-language model) engineered specifically to transcribe dense handwritten university-level mathematics exam sheets directly into compilable, standard LaTeX documents.

It translates raw scanned images into full documents containing equation environments (`align*`, `equation`, `gather`), matrices, fractions, Greek letters, and bracket parity with **82.0% clean PDF compilation rate**.

---

## 🏆 Benchmark Results (Evaluated on 250 Held-Out Test Pages)

v5.0 evaluated on 250 representative held-out test pages from the 700-page split (`test.jsonl`). Metrics include **Character Error Rate (CER)**, **Normalized CER (NCER)**, **BLEU-4**, **chrF**, **Math-F1** symbol score, **Structure Validity %**, and **PDF Compile Rate %**:

| Model | Mean CER ↓ | Median CER ↓ | Norm CER ↓ | BLEU-4 ↑ | chrF ↑ | Math-F1 ↑ | Struct % ↑ | Compile % ↑ | Inference Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base GLM-OCR (0.9B)** | 0.5151 | 0.4609 | 0.4910 | 0.4583 | 0.5370 | 0.7031 | 0.0% | 0.0% | 8.38s |
| **Baidu Unlimited-OCR** | 0.7176 | 0.5768 | 0.7343 | 0.3113 | 0.4792 | 0.6264 | 77.6% | 42.7% | 40.79s |
| **Baidu OCR FT v2** | 0.4258 | 0.3501 | 0.4706 | 0.5958 | 0.7176 | 0.7983 | 71.9% | 64.9% | 29.00s |
| **GLM-OCR Math v3.1** | 0.3971 | 0.2957 | 0.3753 | 0.6180 | 0.7093 | 0.8171 | **91.4%** | **88.9%** | 14.12s |
| **GLM-OCR Math v4.1** | 0.3816 | 0.2783 | 0.4106 | 0.6513 | 0.7531 | 0.8272 | 90.9% | 82.4% | 13.44s |
| **GLM-OCR Math v5.0 (Ours)** | **0.3377** 🏆 | **0.2858** | **0.3683** 🏆 | **0.6594** 🏆 | **0.7539** 🏆 | **0.8358** 🏆 | 85.6% | **82.0%** | **~3.8s (A100)** |

### Key Improvements in v5.0:
- **34.4% Relative CER Reduction** compared to the un-finetuned base model (0.5151 → 0.3377).
- **Lowest Normalized CER (0.3683)** of any tested architecture, eliminating superficial whitespace and macro formatting discrepancies.
- **Record Math-F1 (0.8358)** and **BLEU-4 (0.6594)**.
- **82.0% Compile Pass Rate**: 205 of 250 evaluated documents compile cleanly to PDF out of the box.

---

## ⚙️ Training Specifications

- **Base Model:** `zai-org/GLM-OCR`
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation) via PEFT
- **LoRA Configuration:**
  - Rank ($r$): `32`
  - Alpha: `64`
  - Dropout: `0.05`
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Hardware:** NVIDIA A100-SXM4-80GB (Cloud accelerated)
- **Total Steps:** 3,945 steps
- **Batch Size:** Effective batch size 8 (Batch 1 × Grad Accum 8)
- **Precision:** Native `bfloat16`
- **Optimizer:** AdamW with linear warmup and cosine decay
- **Final Convergence Loss:** `0.0008` (Average epoch training loss: `0.1764`)

---

## 💻 Quickstart Inference Example

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_NAME = "zai-org/GLM-OCR"
ADAPTER_REPO = "realgauravvyas/GLM-OCR-latex"  # Or local path to v5 adapter

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Processor and Base Model in bfloat16
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(device)

# 2. Attach v5.0 LoRA Adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO).to(device)
model.eval()

# 3. Predict LaTeX from handwritten math image
prompt_text = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into a complete, compilable LaTeX document. Ignore printed text, "
    "student info, page numbers, cancelled work and rough work. Output only LaTeX."
)

image = Image.open("sample_math_page.png").convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
latex_code = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

print("Generated LaTeX Document:")
print(latex_code)
```

---

## 📄 License & Attribution
- Licensed under the **MIT License**.
- Built on top of [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR).
- Project repository: [https://github.com/realgauravvyas/ocr2tex](https://github.com/realgauravvyas/ocr2tex)
