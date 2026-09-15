# -*- coding: utf-8 -*-
"""
GLM-OCR v5.0 Training Script - Optimized for Lightning AI Studio (Ubuntu / Cloud GPU)
Hardware Target: NVIDIA L4 (24GB), A10G (24GB), A100 (40GB/80GB), or T4 (16GB)

Features:
- Pure bfloat16 mixed precision
- Ampere TF32 + cuDNN Benchmark
- FlashAttention / SDPA support
- Corrected ETA moving average (eval runs do not pollute training step speed)
- Power-Shield Rolling Checkpoints (every N steps with validation and resume)
"""

import os
import sys
import json
import time
import math
import shutil
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "zai-org/GLM-OCR"
MAX_IMAGE_TOKENS = 1536

USER_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into a complete, compilable LaTeX document. Ignore printed text, "
    "student info, page numbers, cancelled work and rough work. Output only LaTeX."
)


def is_checkpoint_valid(ckpt_dir: Path) -> bool:
    """Check if checkpoint directory has complete, non-corrupted state files."""
    if not ckpt_dir.is_dir():
        return False
    state_file = ckpt_dir / "trainer_state.json"
    if not state_file.exists() or state_file.stat().st_size == 0:
        return False
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False

    has_weights = any(
        (ckpt_dir / w).exists() and (ckpt_dir / w).stat().st_size > 1024 * 512
        for w in ["adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin"]
    )
    if not has_weights:
        return False

    opt_file = ckpt_dir / "optimizer.pt"
    if not opt_file.exists() or opt_file.stat().st_size < 1024 * 512:
        return False

    return True


def find_latest_valid_checkpoint(output_dir: Path):
    """Scans for the latest healthy checkpoint, automatically quarantining corrupted ones."""
    if not output_dir.exists():
        return None

    ckpts = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        reverse=True
    )
    for ckpt in ckpts:
        if is_checkpoint_valid(ckpt):
            return ckpt
        else:
            print(f"\n[POWER-SHIELD] Checkpoint {ckpt.name} is incomplete or corrupted!")
            corrupt_dest = ckpt.with_name(ckpt.name + ".corrupted")
            try:
                if not corrupt_dest.exists():
                    ckpt.rename(corrupt_dest)
                    print(f"   Quarantined to {corrupt_dest.name}. Rolling back...")
            except Exception as e:
                print(f"   Could not rename {ckpt.name}: {e}")
    return None


class MathOCRDataset(Dataset):
    def __init__(self, jsonl_path: Path, image_base: Path):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.image_base = image_base

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.image_base / s["image"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        assistant = next(c["content"] for c in s["conversations"] if c["role"] == "assistant")
        return {"image": image, "target": assistant}


class GlmOcrCollator:
    def __init__(self, processor, max_length=3584, max_image_tokens=1536):
        self.processor = processor
        self.max_length = max_length
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.assistant_id = tok.convert_tokens_to_ids("<|assistant|>")
        self.newline_id = tok.convert_tokens_to_ids("\n")

        ip = processor.image_processor
        patch = getattr(ip, "patch_size", 14)
        merge = getattr(ip, "merge_size", 2)
        pixels_per_token = (patch * patch) * (merge * merge)
        max_pixels = max_image_tokens * pixels_per_token
        try:
            ip.size["longest_edge"] = int(max_pixels)
        except Exception:
            pass

    def __call__(self, batch):
        per_sample = []
        for ex in batch:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_text + ex["target"] + self.processor.tokenizer.eos_token

            full = self.processor(text=[full_text], images=[ex["image"]], return_tensors="pt")
            input_ids = full["input_ids"][0]
            labels = input_ids.clone()

            matches = (input_ids == self.assistant_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                start_idx = matches[-1].item() + 1
                if start_idx < len(input_ids) and input_ids[start_idx].item() == self.newline_id:
                    start_idx += 1
                labels[:start_idx] = -100
            else:
                prompt_only = self.processor(text=[prompt_text], images=[ex["image"]], return_tensors="pt")
                labels[:prompt_only["input_ids"].shape[1]] = -100

            mm_ids = full.get("mm_token_type_ids")
            if mm_ids is not None:
                mm_ids = mm_ids[0]

            if input_ids.shape[0] > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]
                if mm_ids is not None:
                    mm_ids = mm_ids[: self.max_length]

            per_sample.append({"input_ids": input_ids, "labels": labels, "mm_ids": mm_ids, "full": full})

        max_len = max(p["input_ids"].shape[0] for p in per_sample)
        batch_input_ids, batch_labels, batch_attn, batch_mm = [], [], [], []
        for p in per_sample:
            ids = p["input_ids"]
            lbl = p["labels"]
            mm = p["mm_ids"]
            pad = max_len - ids.shape[0]
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), self.pad_id, dtype=ids.dtype)])
                lbl = torch.cat([lbl, torch.full((pad,), -100, dtype=lbl.dtype)])
                if mm is not None:
                    mm = torch.cat([mm, torch.zeros((pad,), dtype=mm.dtype)])
            attn = (ids != self.pad_id).long()
            batch_input_ids.append(ids)
            batch_labels.append(lbl)
            batch_attn.append(attn)
            if mm is not None:
                batch_mm.append(mm)

        out = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attn),
        }
        if batch_mm:
            out["mm_token_type_ids"] = torch.stack(batch_mm)

        pv = [p["full"]["pixel_values"] for p in per_sample]
        out["pixel_values"] = torch.cat(pv, dim=0)
        if "image_grid_thw" in per_sample[0]["full"]:
            grids = [p["full"]["image_grid_thw"] for p in per_sample]
            out["image_grid_thw"] = torch.cat(grids, dim=0)

        return out


class CleanTelemetryCallback(TrainerCallback):
    """Accurately tracks training speed and ETA without pollution from evaluation or disk saving."""
    def __init__(self, metrics_file: Path, meta: dict, save_interval_steps: int = 25):
        self.metrics_file = metrics_file
        self.meta = meta
        self.save_interval_steps = save_interval_steps
        self.start_time = time.time()
        self.step_start_time = None
        self.sec_per_step = None
        self.best_loss = None
        self.best_val_loss = None
        self.history = []
        self.val_history = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start_time is not None:
            step_duration = time.time() - self.step_start_time
            if self.sec_per_step is None:
                self.sec_per_step = step_duration
            else:
                self.sec_per_step = 0.85 * self.sec_per_step + 0.15 * step_duration
            self.step_start_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        elapsed = time.time() - self.start_time
        remaining_steps = max(0, state.max_steps - state.global_step)
        eta_seconds = (remaining_steps * self.sec_per_step) if self.sec_per_step else 0

        loss = logs.get("loss")
        if loss is not None:
            self.best_loss = loss if self.best_loss is None else min(self.best_loss, loss)
            self.history.append({
                "step": state.global_step,
                "loss": round(loss, 4),
                "lr": logs.get("learning_rate"),
                "sec_per_step": round(self.sec_per_step, 2) if self.sec_per_step else None,
            })

        if "eval_loss" in logs:
            vloss = logs["eval_loss"]
            self.best_val_loss = vloss if self.best_val_loss is None else min(self.best_val_loss, vloss)
            self.val_history.append({
                "step": state.global_step,
                "val_loss": round(vloss, 4),
            })

        gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            gpu_mem_gb = round(torch.cuda.max_memory_reserved() / (1024**3), 2)

        eta_str = time.strftime("%Hh %Mm %Ss", time.gmtime(eta_seconds))
        print(
            f"[Step {state.global_step}/{state.max_steps}] "
            f"Loss: {float(loss) if loss is not None else 0.0:.4f} | "
            f"Speed: {self.sec_per_step:.2f}s/step | "
            f"ETA: {eta_str} | "
            f"VRAM: {gpu_mem_gb}GB"
        )

        data = {
            **self.meta,
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": round(state.epoch or 0.0, 3),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta_seconds, 1),
            "sec_per_step": round(self.sec_per_step, 2) if self.sec_per_step else None,
            "best_loss": self.best_loss,
            "best_val_loss": self.best_val_loss,
            "gpu_reserved_gb": gpu_mem_gb,
            "history": self.history[-40:],
            "val_history": self.val_history,
        }
        try:
            tmp = self.metrics_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.metrics_file)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Train GLM-OCR on Lightning AI Studio")
    ap.add_argument("--epochs", type=float, default=2.5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--max-length", type=int, default=3584)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-image-tokens", type=int, default=1536)
    ap.add_argument("--resume", action="store_true", help="Resume from latest valid checkpoint")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-samples", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--dataset-dir", type=str, default="./data")
    ap.add_argument("--image-base", type=str, default="./data")
    ap.add_argument("--output-dir", type=str, default="./output")
    ap.add_argument("--optim", type=str, default="adamw_torch_fused")
    args = ap.parse_args()

    # Hardware Acceleration
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n========================================================")
        print(f"[ACCELERATION] GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"[ACCELERATION] TF32=ON, cuDNN Benchmark=ON, BF16=ON")
        print(f"========================================================\n")
    else:
        print("[WARNING] CUDA not detected. Training will run on CPU (Very slow).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "training_metrics.json"

    print("Loading GLM-OCR processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Loading GLM-OCR model in bfloat16...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "qkv", "proj"
    ]
    print(f"Configuring Dual-Modality LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset_dir = Path(args.dataset_dir)
    image_base = Path(args.image_base)
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Could not find train.jsonl at {train_file.resolve()}")

    train_ds = MathOCRDataset(train_file, image_base)
    val_ds = MathOCRDataset(val_file, image_base)
    print(f"Loaded {len(train_ds)} training samples and {len(val_ds)} validation samples.")

    if args.eval_samples > 0 and len(val_ds) > args.eval_samples:
        val_ds = torch.utils.data.Subset(val_ds, list(range(args.eval_samples)))

    collator = GlmOcrCollator(processor, max_length=args.max_length, max_image_tokens=args.max_image_tokens)

    # Automatically adapt optimizer if fused isn't supported
    chosen_optim = args.optim
    if chosen_optim == "adamw_torch_fused" and not torch.cuda.is_available():
        chosen_optim = "adamw_torch"

    import inspect
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    targs_dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "fp16": False,
        "gradient_checkpointing": True,
        "logging_steps": 5,
        "eval_steps": args.save_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 6,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "dataloader_num_workers": args.workers,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": (args.workers > 0),
        "optim": chosen_optim,
        "remove_unused_columns": False,
    }

    if "warmup_ratio" in sig_params:
        targs_dict["warmup_ratio"] = 0.08
    else:
        targs_dict["warmup_steps"] = 315

    if "eval_strategy" in sig_params:
        targs_dict["eval_strategy"] = "steps"
    elif "evaluation_strategy" in sig_params:
        targs_dict["evaluation_strategy"] = "steps"

    if "save_strategy" in sig_params:
        targs_dict["save_strategy"] = "steps"

    if "gradient_checkpointing_kwargs" in sig_params:
        targs_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    filtered_targs = {k: v for k, v in targs_dict.items() if k in sig_params}
    targs = TrainingArguments(**filtered_targs)

    meta = {
        "model": MODEL_NAME,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }

    telemetry_cb = CleanTelemetryCallback(metrics_file, meta, save_interval_steps=args.save_steps)
    early_cb = EarlyStoppingCallback(early_stopping_patience=4)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[telemetry_cb, early_cb],
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = find_latest_valid_checkpoint(output_dir)
        if resume_ckpt:
            print(f"\n[POWER-SHIELD] Resuming from verified checkpoint: {resume_ckpt.name}\n")
        else:
            print("\n[POWER-SHIELD] No valid prior checkpoint found. Training from scratch.\n")

    print("\nStarting Training on Lightning AI Studio...\n")
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\n[SUCCESS] Training finished! Model and adapter saved to: {final_dir.resolve()}\n")


if __name__ == "__main__":
    main()
