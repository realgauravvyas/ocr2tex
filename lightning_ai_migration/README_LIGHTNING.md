# GLM-OCR v5.0 Training Migration Guide: Local RTX 3060 -> Lightning AI Studio

## Why Migrate?
- **Local RTX 3060 (12 GB):** VRAM is at 111.9%, paging tensors over PCIe to system RAM. Current step time is ~45–75s/step (~30–40 hours remaining).
- **Lightning AI Studio (L4 / A10G 24 GB):** Zero VRAM paging, pure bfloat16, TF32 Tensor Cores, batch size 2. Step time drops to **1.5s – 3.0s/step** (~**2 to 3.5 hours total**, costing ~$2.00 in cloud credits).

---

## Files in this Directory (`D:\ocr2tex\lightning_ai_migration\`)
| File | Purpose |
| :--- | :--- |
| `train_lightning.py` | Optimized training script for Linux/Cloud GPU with corrected ETA telemetry. |
| `run_train.sh` | 1-click launcher script for Linux bash terminal. |
| `requirements.txt` | Python dependencies. |
| `prepare_package.py` | Windows helper to pack code, checkpoint-250, and dataset into compressed `.tar.gz` archives. |
| `upload_to_hf.py` | Alternative 1-click cloud sync to a private Hugging Face dataset repo. |

---

## Step 1: Pack Your Files Locally (Windows)
Open a terminal in `D:\ocr2tex\lightning_ai_migration\` and run:
```powershell
python prepare_package.py
```
This produces three files:
1. `code.tar.gz` (~10 KB) — Contains training scripts.
2. `checkpoint-250.tar.gz` (~160 MB) — Your current best checkpoint so you don't lose any progress.
3. `dataset.tar.gz` (~8-10 GB) — All 14,018 images + train/val/test splits.

---

## Step 2: Create a Studio on Lightning AI
1. Go to **https://lightning.ai** and log in.
2. Click **"New Studio"** (or create one inside your teamspace).
3. In the top-right hardware picker, switch from **CPU** to:
   - **NVIDIA L4 (24 GB)** *(Recommended, best price/performance)*
   - or **NVIDIA A10G (24 GB)**
   - or **NVIDIA A100 (40 GB)**

---

## Step 3: Transfer Data to Lightning AI Studio
Choose **one** of the methods below:

### Option A: Drag & Drop Web UI (Simplest)
- In your Lightning AI Studio browser tab, drag `code.tar.gz`, `checkpoint-250.tar.gz`, and `dataset.tar.gz` into the left file explorer.

### Option B: Cloud Link / Google Drive / gdown (Fastest if uploading via browser is slow)
1. Upload `dataset.tar.gz` to Google Drive or Dropbox.
2. Inside the Lightning AI terminal, download it directly:
   ```bash
   pip install gdown
   gdown "YOUR_GOOGLE_DRIVE_SHARE_LINK" -O dataset.tar.gz
   ```

### Option C: Private Hugging Face Dataset (Enterprise standard)
1. Locally run: `python upload_to_hf.py` (enter your HF repo name).
2. In Lightning AI terminal:
   ```bash
   huggingface-cli login
   git clone https://huggingface.co/datasets/YOUR_USERNAME/YOUR_DATASET data
   ```

---

## Step 4: Extract and Launch Training in Lightning AI
Open the **Terminal** tab in Lightning AI Studio and run:

```bash
# 1. Extract code
tar -xzf code.tar.gz

# 2. Extract dataset (creates ./data folder)
tar -xzf dataset.tar.gz

# 3. (Optional) Extract Checkpoint-250 so you resume from step 250
mkdir -p output
tar -xzf checkpoint-250.tar.gz -C output/

# 4. Make launcher executable and start training
chmod +x run_train.sh
./run_train.sh
```

---

## Step 5: Download Trained Adapter
When training finishes, your fine-tuned LoRA weights are saved in:
`./output/final`
You can right-click the `final` folder in the Lightning AI file explorer and select **Download**, or compress it:
```bash
tar -czf final_model.tar.gz -C output final
```
