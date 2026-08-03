"""EXPERIMENT: LoRA fine-tuning for Baidu Unlimited-OCR, v2 (fixed hyperparams).

v2 fixes relative to the original (D:\\Claude Code\\BaiduOCR\\train_unlimited_ocr.py):
  1. --max-target-len default 1024 -> 2048  (p99 of train targets = 1256, so at
     1024 ~5% of pages had their \\end{document} cut off and EOS appended,
     teaching the model to truncate early).
  2. Added linear LR warmup before the cosine decay (cold start made the first
     ~200 steps unnecessarily noisy).
  3. --early-stopping-patience default 0 -> 3  (run used the FINAL adapter after
     LR decayed to ~0; with patience>0 and best-checkpoint saving we keep the
     lowest-val weights instead of whatever the cosine leaves at the end).

Everything else (dataset, example construction, forward-loss wiring, resume
logic) is unchanged so the run is directly comparable to v1. Original file is
untouched; this is a standalone experiment copy. Writes to a NEW output dir so
the previous baidu-ocr-math-v1 finetune and benchmark column are never clobbered.
"""
import os, sys, json, time, math, argparse, random
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # anti-fragmentation
from pathlib import Path
from PIL import ImageOps
import torch

MODEL_NAME = "baidu/Unlimited-OCR"
IMAGE_TOKEN_ID = 128815
LORA_TARGETS = ["q_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj"]


def patch_tf():
    try:
        from transformers.utils.import_utils import is_torch_fx_available  # noqa
    except ImportError:
        import transformers.utils.import_utils as _i, transformers.utils as _t
        _i.is_torch_fx_available = _t.is_torch_fx_available = lambda: False


def load_model(qlora):
    patch_tf()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    kw = dict(trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16)
    if qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModel.from_pretrained(MODEL_NAME, **kw)
    return model, tok


def helpers(model):
    """The model's own preprocessing functions, from its trust_remote_code module."""
    mod = sys.modules[type(model).__module__]
    return (mod.format_messages, mod.load_pil_images, mod.text_encode,
            mod.dynamic_preprocess, mod.BasicImageTransform)


def build_example(M, tok, image_path, prompt, target, base_size, image_size, max_target_len, max_crops=16):
    """Replicates infer()'s input construction, then appends target + builds labels."""
    fmt, load_imgs, t_enc, dyn, Tfm = M
    conv = [{"role": "<|User|>", "content": prompt, "images": [str(image_path)]},
            {"role": "<|Assistant|>", "content": ""}]
    prompt_str = fmt(conversations=conv, sft_format="plain", system_prompt="")
    images = load_imgs(conv)
    tfm = Tfm(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    patch_size, ds = 16, 4
    splits = prompt_str.split("<image>")
    imgs_global, imgs_crop, seq_mask, ids, spatial = [], [], [], [], []
    for text_sep, image in zip(splits, images):
        t = t_enc(tok, text_sep, bos=False, eos=False)
        ids += t; seq_mask += [False] * len(t)
        if image.size[0] <= 640 and image.size[1] <= 640:
            crop_ratio, crop_raw = [1, 1], []
        else:
            crop_raw, crop_ratio = dyn(image, max_num=max_crops)  # cap crops -> bound seq len / VRAM
        gv = ImageOps.pad(image, (base_size, base_size), color=tuple(int(x * 255) for x in tfm.mean))
        imgs_global.append(tfm(gv).to(torch.bfloat16))
        wc, hc = crop_ratio; spatial.append([wc, hc])
        if wc > 1 or hc > 1:
            for ci in crop_raw:
                imgs_crop.append(tfm(ci).to(torch.bfloat16))
        nq = math.ceil((image_size // patch_size) / ds)
        nqb = math.ceil((base_size // patch_size) / ds)
        timg = ([IMAGE_TOKEN_ID] * nqb + [IMAGE_TOKEN_ID]) * nqb + [IMAGE_TOKEN_ID]
        if wc > 1 or hc > 1:
            timg += ([IMAGE_TOKEN_ID] * (nq * wc) + [IMAGE_TOKEN_ID]) * (nq * hc)
        ids += timg; seq_mask += [True] * len(timg)
    t = t_enc(tok, splits[-1], bos=False, eos=False)
    ids += t; seq_mask += [False] * len(t)
    ids = [0] + ids; seq_mask = [False] + seq_mask          # bos
    prompt_len = len(ids)
    tgt = t_enc(tok, target, bos=False, eos=False)[:max_target_len] + [tok.eos_token_id]
    ids += tgt; seq_mask += [False] * len(tgt)
    input_ids = torch.LongTensor(ids)
    seq_mask = torch.tensor(seq_mask, dtype=torch.bool)
    labels = torch.LongTensor([-100] * prompt_len + tgt)
    images_ori = torch.stack(imgs_global, 0)
    spatial = torch.tensor(spatial, dtype=torch.long)
    images_crop = torch.stack(imgs_crop, 0) if imgs_crop else torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16)
    return input_ids, seq_mask, labels, images_crop, images_ori, spatial


def forward_loss(model, ex, dev):
    input_ids, seq_mask, labels, images_crop, images_ori, spatial = ex
    out = model(
        input_ids=input_ids.unsqueeze(0).to(dev),
        attention_mask=torch.ones((1, input_ids.shape[0]), dtype=torch.long, device=dev),
        images=[(images_crop.to(dev, torch.bfloat16), images_ori.to(dev, torch.bfloat16))],
        images_seq_mask=seq_mask.unsqueeze(0).to(dev),
        images_spatial_crop=spatial.to(dev),
        labels=labels.unsqueeze(0).to(dev),
        use_cache=False,   # training: no KV cache (also required for grad checkpointing)
    )
    return out.loss


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def target_of(row):
    return next((c["content"] for c in row["conversations"] if c["role"] in ("assistant", "gpt")), "")


def write_metrics(path, **kw):
    try:
        cur = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
        cur.update(kw); cur["updated_at"] = time.time()
        Path(path).write_text(json.dumps(cur), encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    # Dataset is READ-ONLY (shared with GLM-OCR, never written). Outputs go to a
    # dedicated Baidu folder so GLM-OCR's models/data are never touched.
    ap.add_argument("--train-file", default=r"D:\ocr2tex\workspace\9_split\train.jsonl")
    ap.add_argument("--val-file", default=r"D:\ocr2tex\workspace\9_split\val.jsonl")
    ap.add_argument("--image-base", default=r"D:\ocr2tex\workspace\9_split")
    ap.add_argument("--output-dir", default=r"D:\ocr2tex\experiments\baidu_v2\out\baidu-ocr-math-v2")
    ap.add_argument("--prompt", default="<image>Convert the handwriting to a complete LaTeX document.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-target-len", type=int, default=2048, help="v2: 2048 covers p99=1256; 1024 truncated ~5% of pages mid-document")
    ap.add_argument("--max-crops", type=int, default=16, help="cap image tiles per page (bounds VRAM)")
    ap.add_argument("--base-size", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--eval-samples", type=int, default=80)
    ap.add_argument("--early-stopping-patience", type=int, default=3, help="stop if val loss doesn't improve for N evals (0=off). v2 default 3: keep the best checkpoint instead of whatever the cosine leaves at the end.")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    ap.add_argument("--warmup-steps", type=int, default=0, help="v2: linear LR warmup steps before cosine (0=auto=10%% of total)")
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume from <output-dir>/checkpoint")
    ap.add_argument("--smoke", action="store_true", help="build 1 example + 1 forward, print loss, exit")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "training_metrics.json"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    write_metrics(metrics_path, status="loading", started_at=time.time(), history=[])

    print("loading model...", flush=True)
    model, tok = load_model(args.qlora)
    M = helpers(model)
    base = Path(args.image_base)

    def to_example(row):
        return build_example(M, tok, base / row["image"], args.prompt, target_of(row),
                             args.base_size, args.image_size, args.max_target_len, args.max_crops)

    # ---- smoke test: one example, one forward (validate the whole pipeline) ----
    if args.smoke:
        model.to(dev).train()
        row = read_jsonl(args.train_file)[0]
        ex = to_example(row)
        print(f"input_ids={ex[0].shape} seq_mask_true={int(ex[1].sum())} "
              f"labels_supervised={int((ex[2]!=-100).sum())} crop={ex[3].shape} global={ex[4].shape} spatial={ex[5].tolist()}", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = forward_loss(model, ex, dev)
        print(f"SMOKE OK: loss={loss.item():.4f}", flush=True)
        write_metrics(metrics_path, status="smoke_ok", smoke_loss=round(loss.item(), 4))
        return

    # ---- LoRA (fresh, or resumed from a saved checkpoint) ----
    from peft import LoraConfig, get_peft_model, PeftModel
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    ckpt = out / "checkpoint"
    resuming = args.resume and (ckpt / "adapter_config.json").exists()
    if resuming:
        model = PeftModel.from_pretrained(model, str(ckpt), is_trainable=True)
        print(f"RESUMING adapter from {ckpt}", flush=True)
    else:
        lconf = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                           target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lconf)
    model.to(dev)
    # NOTE: gradient checkpointing is intentionally OFF. It needs
    # enable_input_require_grads(), which makes inputs_embeds require grad; the model
    # then splices image features with an in-place masked_scatter_ on a view of
    # inputs_embeds, which PyTorch forbids on a grad-requiring leaf view. We rely on
    # bf16 + LoRA + batch 1 (use --qlora if it OOMs on 12GB).
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.2f}M", flush=True)

    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)[: args.eval_samples]
    random.seed(0)
    steps_per_epoch = max(1, len(train_rows) // args.grad_accum)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # v2 fix: linear warmup -> cosine decay (cold start made the first ~200 steps noisy).
    # Warmup covers the first `warmup_steps` optimizer steps; cosine then decays the rest.
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(20, int(0.10 * total_steps))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))
    warm = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))
    sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[warmup_steps])
    print(f"LR schedule: {warmup_steps} warmup steps -> cosine over {total_steps - warmup_steps}", flush=True)
    start_step, resume_best, resume_hist = 0, None, []
    rs = ckpt / "resume_state.pt"
    if resuming and rs.exists():
        try:
            st = torch.load(rs, map_location=dev)
            opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            start_step = int(st.get("step", 0)); resume_best = st.get("best_val"); resume_hist = st.get("history", [])
            print(f"RESUMED optimizer at step {start_step}", flush=True)
        except Exception as e:
            print(f"could not load resume_state ({e}) - continuing with restored weights only", flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval(); losses = []
        for row in val_rows:
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses.append(forward_loss(model, to_example(row), dev).item())
            except Exception:
                pass
        model.train()
        return sum(losses) / len(losses) if losses else None

    write_metrics(metrics_path, status="running", step=0, max_steps=total_steps,
                  epochs=args.epochs, trainable_m=round(trainable/1e6, 2), history=[])
    print(f"training: {len(train_rows)} samples, {total_steps} steps "
          f"(accum {args.grad_accum}, ~{args.epochs} epochs)", flush=True)

    step, t0, best_val, micro, hist = start_step, time.time(), resume_best, 0, list(resume_hist)
    fails = 0  # consecutive sample failures -> abort if systemic (don't spin for hours)
    no_improve = 0  # consecutive evals without a val-loss improvement (early stopping)
    aborted = False
    opt.zero_grad()
    stop = step >= total_steps
    for epoch in range(math.ceil(args.epochs)):
        order = list(range(len(train_rows))); random.shuffle(order)
        for ri in order:
            try:
                ex = to_example(train_rows[ri])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = forward_loss(model, ex, dev) / args.grad_accum
                loss.backward()
                fails = 0
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt.zero_grad(); micro = 0; fails += 1
                print("  OOM on a sample - skipped", flush=True)
                if fails >= 80:
                    write_metrics(metrics_path, status="error", error="too many consecutive OOMs - try --qlora or lower --max-crops")
                    print("ABORT: 80 consecutive OOMs", flush=True); aborted = True; stop = True; break
                continue
            except Exception as e:
                fails += 1
                print(f"  sample error: {type(e).__name__}: {str(e)[:80]}", flush=True)
                if fails >= 60:
                    write_metrics(metrics_path, status="error", error=f"systemic error: {type(e).__name__}: {str(e)[:120]}")
                    print("ABORT: 60 consecutive sample errors", flush=True); aborted = True; stop = True; break
                continue
            micro += 1
            if micro < args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(); micro = 0
            step += 1
            cur_loss = loss.item() * args.grad_accum
            # write metrics EVERY step so the dashboard updates ~every 46s (not every
            # 5 steps) — a resume starting mid-run otherwise looks frozen for minutes
            el = time.time() - t0; eta = el / max(1, step - start_step) * (total_steps - step)
            if step % 5 == 0 or step <= start_step + 2:
                hist = (hist + [{"step": step, "loss": round(cur_loss, 4)}])[-400:]
            write_metrics(metrics_path, status="running", step=step, max_steps=total_steps,
                          loss=round(cur_loss, 4), lr=sched.get_last_lr()[0],
                          percent=round(100*step/total_steps, 1), eta_s=int(eta),
                          epoch=epoch, history=hist, best_val=best_val)
            if step % 5 == 0 or step <= start_step + 2:
                print(f"  step {step}/{total_steps} loss={cur_loss:.4f} eta={eta/60:.0f}m", flush=True)
            if step % args.eval_steps == 0:
                vl = evaluate()
                if vl is not None:
                    if best_val is None or vl < best_val - 1e-4:
                        best_val = vl; no_improve = 0
                        model.save_pretrained(str(out / "best")); print(f"  saved best (val {vl:.4f})", flush=True)
                    else:
                        no_improve += 1
                        print(f"  val {vl:.4f} (no improvement {no_improve}/{args.early_stopping_patience or '-'})", flush=True)
                    hist = (hist + [{"step": step, "val_loss": round(vl, 4)}])[-400:]
                    write_metrics(metrics_path, status="running", step=step, val_loss=round(vl, 4),
                                  best_val=round(best_val, 4), history=hist)
                    if args.early_stopping_patience and no_improve >= args.early_stopping_patience:
                        print(f"EARLY STOP: val loss not improved for {no_improve} evals — best={best_val:.4f}", flush=True)
                        stop = True; break
            if step % args.save_steps == 0:
                model.save_pretrained(str(ckpt))
                torch.save({"step": step, "best_val": best_val, "history": hist,
                            "opt": opt.state_dict(), "sched": sched.state_dict()}, ckpt / "resume_state.pt")
            if step >= total_steps:
                stop = True; break
        if stop:
            break

    if aborted:
        print("TRAINING ABORTED — status=error kept; not saving 'final'.", flush=True)
    else:
        model.save_pretrained(str(out / "final"))
        write_metrics(metrics_path, status="done", step=step, max_steps=total_steps,
                      best_val=best_val, finished_at=time.time())
        print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
