"""Fine-tune GLM-OCR (0.9B) on the OCR2TeX merged dataset (LoRA, RTX 3060 12GB).

Adapted from the proven 06c_finetune_v3.2.py recipe with the new dataset:
- dataset: D:\\ocr2tex\\workspace\\9_split (12,575 train / 698 val samples)
- epochs default 2 (dataset is 2.2x larger than v3's 5,840 samples, so 2 epochs
  gives ~the same optimizer-update count that converged for v3/v4 in 4-5 epochs,
  at half the wall time)
- early stopping (patience 3 evals) + load_best_model_at_end guard the single
  training attempt against overfitting
- everything else (LoRA config, collator, loss masking, metrics JSON) is the
  v3.2 recipe unchanged

Launched by the dashboard (app.py) as a subprocess; live metrics are written to
<output-dir>/training_metrics.json.
"""

import os
import json
import time
import math
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_NAME = "zai-org/GLM-OCR"
DATASET_DIR = Path(r"D:\ocr2tex\workspace\9_split")
OUTPUT_DIR = Path(r"D:\ocr2tex\output\glm-ocr-math-v4")

MAX_IMAGE_TOKENS = 1536

USER_PROMPT = (
    "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
    "content into a complete, compilable LaTeX document. Ignore printed text, "
    "student info, page numbers, cancelled work and rough work. Output only LaTeX."
)


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
        except Exception as e:
            print(f"WARNING: Could not load {img_path}: {e}. Using blank image.")
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        assistant = next(c["content"] for c in s["conversations"] if c["role"] == "assistant")
        return {"image": image, "target": assistant}


class GlmOcrCollator:
    def __init__(self, processor, max_length=3584, max_image_tokens=1536):
        self.processor = processor
        self.max_length = max_length
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        # Cap image resolution: tokens ~= pixels / (patch^2 * merge^2) = pixels / 784
        ip = processor.image_processor
        patch = getattr(ip, "patch_size", 14)
        merge = getattr(ip, "merge_size", 2)
        pixels_per_token = (patch * patch) * (merge * merge)
        max_pixels = max_image_tokens * pixels_per_token
        try:
            ip.size["longest_edge"] = int(max_pixels)
            self.max_pixels = max_pixels
        except Exception:
            self.max_pixels = None

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
            prompt_only = self.processor(text=[prompt_text], images=[ex["image"]], return_tensors="pt")

            input_ids = full["input_ids"][0]
            prompt_len = prompt_only["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:prompt_len] = -100  # learn only the LaTeX answer

            mm_ids = full.get("mm_token_type_ids")
            if mm_ids is not None:
                mm_ids = mm_ids[0]

            # End-truncation never cuts image placeholders (they sit at the front)
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


def _safe_num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isinf(f) or math.isnan(f):
        return None
    return f


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    return obj


class DashboardCallback(TrainerCallback):
    def __init__(self, metrics_file: Path, meta: dict):
        self.metrics_file = metrics_file
        self.meta = meta
        self.history = []
        self.val_history = []
        self.start = time.time()
        self.last_step_time = None
        self.last_step = 0
        self.best_loss = None
        self.best_val_loss = None
        self.sec_per_step = None
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self._write(state="starting")

    def _gpu_stats(self):
        if not torch.cuda.is_available():
            return None, None, None
        try:
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return round(alloc, 2), round(reserved, 2), round(total, 2)
        except Exception:
            return None, None, None

    def _write(self, state="running", global_step=0, epoch=0.0, total_steps=0):
        elapsed = time.time() - self.start
        eta = None
        if global_step > 0 and total_steps > 0 and self.sec_per_step:
            eta = self.sec_per_step * (total_steps - global_step)
        elif global_step > 0 and total_steps > 0:
            eta = elapsed / global_step * (total_steps - global_step)

        gpu_alloc, gpu_reserved, gpu_total = self._gpu_stats()
        pct = (100.0 * global_step / total_steps) if total_steps else 0.0

        trend = None
        losses = [h["loss"] for h in self.history if h["loss"] is not None]
        if len(losses) >= 6:
            recent = sum(losses[-3:]) / 3
            older = sum(losses[-6:-3]) / 3
            trend = "down" if recent < older else ("up" if recent > older else "flat")

        data = {
            **self.meta,
            "state": state,
            "global_step": global_step,
            "total_steps": total_steps,
            "epoch": epoch,
            "progress_pct": round(pct, 2),
            "elapsed": elapsed,
            "eta": eta,
            "sec_per_step": self.sec_per_step,
            "steps_per_sec": (1.0 / self.sec_per_step) if self.sec_per_step else None,
            "gpu_alloc_gb": gpu_alloc,
            "gpu_reserved_gb": gpu_reserved,
            "gpu_total_gb": gpu_total,
            "gpu_pct": round(100.0 * gpu_reserved / gpu_total, 1) if (gpu_reserved and gpu_total) else None,
            "best_loss": self.best_loss,
            "best_val_loss": self.best_val_loss,
            "loss_trend": trend,
            "history": self.history[-1000:],
            "val_history": self.val_history[-300:],
            "last_loss": losses[-1] if losses else None,
            "last_lr": self.history[-1]["lr"] if self.history else None,
            "last_grad": self.history[-1]["grad_norm"] if self.history else None,
            "last_val_loss": self.val_history[-1]["val_loss"] if self.val_history else None,
        }
        tmp = self.metrics_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(_json_safe(data), allow_nan=False), encoding="utf-8")
        tmp.replace(self.metrics_file)

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self.last_step_time is not None and state.global_step > self.last_step:
            dt = (now - self.last_step_time) / max(1, (state.global_step - self.last_step))
            self.sec_per_step = dt if self.sec_per_step is None else 0.8 * self.sec_per_step + 0.2 * dt
        self.last_step_time = now
        self.last_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            loss = _safe_num(logs.get("loss"))
            _, gpu_reserved, _ = self._gpu_stats()
            self.history.append({
                "step": state.global_step,
                "epoch": round(state.epoch or 0.0, 3),
                "loss": loss,
                "lr": _safe_num(logs.get("learning_rate")),
                "grad_norm": _safe_num(logs.get("grad_norm")),
                "gpu_mem": gpu_reserved,
                "sec_per_step": _safe_num(self.sec_per_step),
            })
            if loss is not None:
                self.best_loss = loss if self.best_loss is None else min(self.best_loss, loss)
        if "eval_loss" in logs:
            vloss = _safe_num(logs["eval_loss"])
            self.val_history.append({
                "step": state.global_step,
                "val_loss": vloss,
                "eval_runtime": _safe_num(logs.get("eval_runtime")),
                "eval_sps": _safe_num(logs.get("eval_samples_per_second")),
            })
            if vloss is not None:
                self.best_val_loss = vloss if self.best_val_loss is None else min(self.best_val_loss, vloss)
        self._write(state="running",
                    global_step=state.global_step,
                    epoch=state.epoch or 0.0,
                    total_steps=state.max_steps)

    def on_train_end(self, args, state, control, **kwargs):
        self._write(state="done", global_step=state.global_step,
                    epoch=state.epoch or 0.0, total_steps=state.max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2,
                    help="2 epochs x 12,575 samples ~= the update count that converged for v3/v4 on 5,840 samples")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=3584)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-image-tokens", type=int, default=1536)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in --output-dir")
    ap.add_argument("--warm-start-adapter", type=str, default="",
                    help="Path to a previous LoRA adapter to warm-start from (optional)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-samples", type=int, default=200)
    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    # dataset overrides (v4.1 uses corrected jsonl + shared 9_split images)
    ap.add_argument("--train-file", type=str, default="")
    ap.add_argument("--val-file", type=str, default="")
    ap.add_argument("--image-base", type=str, default="")
    # speed / memory optimizations
    ap.add_argument("--optim", type=str, default="adamw_torch",
                    help="adamw_torch | paged_adamw_8bit (bitsandbytes 8-bit optimizer, frees VRAM)")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable gradient checkpointing (~30%% faster, needs more VRAM)")
    ap.add_argument("--qlora", action="store_true",
                    help="load base model in 4-bit (QLoRA) to free VRAM for bigger batch / no checkpointing")
    ap.add_argument("--tf32", action="store_true", help="enable TF32 matmul on Ampere (free speedup)")
    ap.add_argument("--save-steps", type=int, default=0,
                    help="checkpoint every N steps (0=auto, ~1/4 epoch). Smaller = lose less on a crash/shutdown")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="sleep N seconds after each step to keep the PC usable during training (slows training)")
    args = ap.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    metrics_file = output_dir / "training_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    load_kwargs = dict(trust_remote_code=True, dtype=torch.bfloat16)
    if args.qlora:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        print("Loading model (QLoRA 4-bit)...")
    else:
        print("Loading model (bf16)...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, **load_kwargs)
    model.config.use_cache = False
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_grad_checkpoint)

    warm = Path(args.warm_start_adapter) if args.warm_start_adapter else None
    if warm and warm.exists():
        print(f"Warm-starting from adapter: {warm}")
        model = PeftModel.from_pretrained(model, str(warm), is_trainable=True)
    else:
        if warm:
            print(f"WARNING: warm-start adapter {warm} not found; using fresh LoRA")
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)

    model.print_trainable_parameters()
    if not args.qlora:
        model.to(device)  # 4-bit model is already placed by from_pretrained
    model.enable_input_require_grads()
    grad_ckpt = not args.no_grad_checkpoint
    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    image_base = Path(args.image_base) if args.image_base else DATASET_DIR
    train_file = Path(args.train_file) if args.train_file else DATASET_DIR / "train.jsonl"
    val_file = Path(args.val_file) if args.val_file else DATASET_DIR / "val.jsonl"
    train_ds = MathOCRDataset(train_file, image_base)
    val_ds = MathOCRDataset(val_file, image_base)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  (images: {image_base})")

    if args.eval_samples > 0 and len(val_ds) > args.eval_samples:
        val_ds = torch.utils.data.Subset(val_ds, list(range(args.eval_samples)))
        print(f"Eval subset: using first {args.eval_samples} val samples for speed")

    collator = GlmOcrCollator(processor, max_length=args.max_length, max_image_tokens=args.max_image_tokens)

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    eval_save_steps = args.save_steps if args.save_steps > 0 else max(50, steps_per_epoch // 4)

    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        gradient_checkpointing=grad_ckpt,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=(args.workers > 0),
        remove_unused_columns=False,
        optim=args.optim,
        seed=args.seed,
    )

    meta = {
        "model": MODEL_NAME,
        "pid": os.getpid(),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_length": args.max_length,
        "max_image_tokens": args.max_image_tokens,
        "seed": args.seed,
        "warm_start": bool(warm and warm.exists()),
        "steps_per_epoch": steps_per_epoch,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    dash_cb = DashboardCallback(metrics_file, meta)
    callbacks = [dash_cb]
    if args.throttle > 0:
        class _Throttle(TrainerCallback):
            def on_step_end(self, a, s, c, **k):
                time.sleep(args.throttle)
        callbacks.append(_Throttle())
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.0,
        ))

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    print(f"\nTotal optimizer steps: {total_steps}")
    print(f"Dashboard metrics -> {metrics_file}")
    print("Starting training...\n")

    resume = None
    if args.resume:
        ckpts = list(output_dir.glob("checkpoint-*"))
        if ckpts:
            def step_num(p):
                try:
                    return int(p.name.split("-")[-1])
                except ValueError:
                    return -1
            latest = max(ckpts, key=step_num)
            resume = str(latest)
            print(f"Resuming from: {resume}")
        else:
            print("WARNING: --resume specified but no checkpoint found. Starting fresh.")

    trainer.train(resume_from_checkpoint=resume)

    print("\nSaving final LoRA adapter...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Done. Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
