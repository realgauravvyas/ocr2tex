#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "GLM-OCR v5.0 Training - Lightning AI Studio Launcher"
echo "=========================================================="

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found! Please ensure your Studio is running with GPU (A100 or L4)."
    exit 1
fi

nvidia-smi

# Detect GPU type
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
echo ""
echo "[HARDWARE DETECTED] $GPU_NAME"

# Set CUDA optimization environment variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Verify requirements
echo "Checking Python dependencies..."
pip install -r requirements.txt --quiet

# Check dataset structure
if [ ! -f "./data/train.jsonl" ]; then
    echo "[ERROR] ./data/train.jsonl not found!"
    echo "Please ensure you have extracted dataset.tar.gz into the studio (so ./data exists)."
    exit 1
fi

# Auto-configure based on GPU
if [[ "$GPU_NAME" =~ "A100" ]]; then
    echo "[PROFILE] Applying A100 High-Throughput Profile (Batch 1, Accum 8, Workers 6)"
    BATCH_SIZE=1
    GRAD_ACCUM=8
    WORKERS=6
    OPTIM="adamw_torch_fused"
elif [[ "$GPU_NAME" =~ "L4" ]]; then
    echo "[PROFILE] Applying L4 Balanced Profile (Batch 2, Accum 4, Workers 4)"
    BATCH_SIZE=2
    GRAD_ACCUM=4
    WORKERS=4
    OPTIM="adamw_torch_fused"
else
    echo "[PROFILE] Applying Standard Profile (Batch 1, Accum 8, Workers 2)"
    BATCH_SIZE=1
    GRAD_ACCUM=8
    WORKERS=2
    OPTIM="adamw_torch_fused"
fi

echo "=========================================================="
echo "Starting Training from Checkpoint (Effective Batch 8)..."
echo "=========================================================="

TRAIN_START=$(date +%s)

python train_lightning.py \
  --epochs 2.5 \
  --batch-size $BATCH_SIZE \
  --grad-accum $GRAD_ACCUM \
  --lr 2.5e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --save-steps 50 \
  --resume \
  --optim $OPTIM \
  --dataset-dir ./data \
  --image-base ./data \
  --output-dir ./output \
  --workers $WORKERS

TRAIN_END=$(date +%s)
TRAIN_DUR=$((TRAIN_END - TRAIN_START))

echo ""
echo "=========================================================="
echo "[COMPLETE] Training successfully finished in $((TRAIN_DUR / 60)) minutes!"
echo "Adapter weights saved in: ./output/final"
echo "=========================================================="

BENCH_DUR=0
if [ -f "benchmark_lightning.py" ] && [ -f "./data/test.jsonl" ]; then
    echo ""
    echo "=========================================================="
    echo "Running Official 700-Page Benchmark on Held-Out Test Set..."
    echo "=========================================================="
    BENCH_START=$(date +%s)
    python benchmark_lightning.py --dataset-dir ./data --image-base ./data --output-dir ./output
    BENCH_END=$(date +%s)
    BENCH_DUR=$((BENCH_END - BENCH_START))
fi

# Generate Executive Summary Report
if [ -f "generate_summary_report.py" ]; then
    python generate_summary_report.py "$TRAIN_DUR" "$BENCH_DUR" "$GPU_NAME"
fi

# Stage and package the complete bundle
echo ""
echo "=========================================================="
echo "Packaging Complete Deliverable (Model + Metrics + Report)..."
echo "=========================================================="

mkdir -p package_stage/final_package
if [ -d "./output/final" ]; then
    cp -r ./output/final package_stage/final_package/adapter
else
    LATEST_CKPT=$(ls -td ./output/checkpoint-* 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CKPT" ]; then
        cp -r "$LATEST_CKPT" package_stage/final_package/adapter
    fi
fi

if [ -f "./output/TRAINING_AND_BENCHMARK_REPORT.md" ]; then
    cp ./output/TRAINING_AND_BENCHMARK_REPORT.md package_stage/final_package/
fi
if [ -f "./output/benchmark_results_v5.json" ]; then
    cp ./output/benchmark_results_v5.json package_stage/final_package/
fi
if [ -f "./output/training_metrics.json" ]; then
    cp ./output/training_metrics.json package_stage/final_package/
fi
if [ -f "./output/trainer_state.json" ]; then
    cp ./output/trainer_state.json package_stage/final_package/
fi
if [ -d "./output/bench_outputs" ]; then
    cp -r ./output/bench_outputs package_stage/final_package/
fi

tar -czf final_results_package.tar.gz -C package_stage final_package
rm -rf package_stage

if [ -d "./output/final" ]; then
    tar -czf final_model.tar.gz -C output final
fi

echo ""
echo "=========================================================="
echo "[SUCCESS] ALL RESULTS PACKAGED IN: final_results_package.tar.gz"
echo "Contains:"
echo "  1. adapter/ (Fine-tuned LoRA weights)"
echo "  2. TRAINING_AND_BENCHMARK_REPORT.md (Executive scorecard)"
echo "  3. benchmark_results_v5.json (Metrics across all 700 test pages)"
echo "  4. training_metrics.json & trainer_state.json (Loss curves & parameters)"
echo "  5. bench_outputs/ (700 generated LaTeX files)"
echo ""
echo "To download: In the left file explorer, right-click 'final_results_package.tar.gz' and click 'Download'."
echo "=========================================================="

# Cost-protection: Attempt auto-stop to prevent idle GPU credit burn
echo "[IDLE PROTECTION] Requesting Studio shutdown to save credits..."
lightning studio stop 2>/dev/null || true
