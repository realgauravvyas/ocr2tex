import re
from pathlib import Path

fpath = Path(r"D:\ocr2tex\lightning_ai_migration\train_lightning.py")
content = fpath.read_text(encoding="utf-8")

pattern = r"    targs = TrainingArguments\(.*?\n    \)"
replacement = """    import inspect
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
    targs = TrainingArguments(**filtered_targs)"""

new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
if count > 0:
    fpath.write_text(new_content, encoding="utf-8")
    print(f"SUCCESS: replaced {count} occurrence(s)")
else:
    print("FAILED to find pattern")
