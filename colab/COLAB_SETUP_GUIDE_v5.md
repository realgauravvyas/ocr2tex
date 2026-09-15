# GLM-OCR v5.0 Colab Setup & Deployment Guide

## 🚀 One-Time Setup (Before Sharing the Notebook)

### Step 1 — Verify the Package Zip
We have already built `glm_ocr_v5_package.zip` (660.1 MB) in `D:\ocr2tex\colab\`.
It contains:
- `adapter/` — GLM-OCR v5.0 fine-tuned LoRA weights & tokenizer configs
- `test_manifest.json` — Ground truth LaTeX and metadata for all 700 held-out test pages
- `test_images/` — Scanned test images for all 250 pages (held-out test split) (pages 1 to 700)

*(To rebuild in the future, run: `python prepare_colab_package_v5.py`)*

---

### Step 2 — Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com).
2. Upload `D:\ocr2tex\colab\glm_ocr_v5_package.zip`.
3. Right-click the uploaded zip file → **Share** → Set to **'Anyone with the link can view'** (Viewer).
4. Copy the link. The file ID is the string between `/d/` and `/view`:
   ```
   https://drive.google.com/file/d/1GoJNcUvE2thjtnsverSNbZwG1gzBPqXf/view
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                           (This is your FILE_ID)
   ```

---

### Step 3 — Paste the File ID into `GLM_ocr_v5_Demo.ipynb`

Open `D:\ocr2tex\colab\notebooks\GLM_ocr_v5_Demo.ipynb`, find Step 3:
```python
ADAPTER_DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
```
Replace `"YOUR_GOOGLE_DRIVE_FILE_ID_HERE"` with your copied file ID:
```python
ADAPTER_DRIVE_FILE_ID = "1GoJNcUvE2thjtnsverSNbZwG1gzBPqXf"
```

---

### Step 4 — Open in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File → Upload notebook** and upload `GLM_ocr_v5_Demo.ipynb`.
3. Set runtime to **GPU (T4)**: **Runtime → Change runtime type → T4 GPU → Save**.
4. Run all cells sequentially!

---

## 🎯 What the Interactive v5.0 Notebook Provides

1. **Test Any Page (1 to 700):**
   - Enter any number from 1 to 700 to run on that exact held-out test page.
   - Shows the original handwritten scan, ground-truth reference LaTeX, and v5.0 compiled PDF.
2. **Random Test Mode:**
   - Automatically draws a random page from the 700 test set.
3. **Custom Upload:**
   - Drag-and-drop or upload your own math notes to transcribe and compile to PDF.
4. **Dual-Model Comparison:**
   - Toggle `Compare_With_Base = True` to see Base GLM-OCR (0.9B) vs GLM-OCR v5.0 side-by-side.
5. **Scorecard Summary:**
   - Computes live Levenshtein CER against ground truth.

---

## 🏆 Benchmark Summary (v5.0 vs Older Models)

| Metric | Base GLM-OCR | v3.1 | v4.1 | **v5.0 (SOTA)** |
|---|:---:|:---:|:---:|:---:|
| **Mean CER (↓)** | 0.5151 | 0.3971 | 0.3816 | **0.3377** 🏆 |
| **Normalized CER (↓)** | 0.4910 | 0.3753 | 0.4106 | **0.3683** 🏆 |
| **BLEU-4 (↑)** | 0.4583 | 0.6180 | 0.6513 | **0.6594** 🏆 |
| **Math-F1 (↑)** | 0.7031 | 0.8171 | 0.8272 | **0.8358** 🏆 |
| **Compile Rate (↑)** | 0.0% | 88.9% | 82.4% | **82.0%** |
