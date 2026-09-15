#!/usr/bin/env bash
echo "=========================================================="
echo "GLM-OCR v5.0 Independent Benchmark Launcher"
echo "=========================================================="
python benchmark_lightning.py \
  --dataset-dir ./data \
  --image-base ./data \
  --output-dir ./output \
  --limit ${1:-100}
