r"""LoRA fine-tuning for Baidu Unlimited-OCR, v3.

Isolated experiment: reads D:\ocr2tex\workspace\9_split read-only, writes ONLY
under D:\Claude Code\BaiduOCR-v3\out. The v1 trainer, the v2 experiment copy and
every existing benchmark artifact are left untouched.

Changes vs v1/v2, in descending expected impact
-----------------------------------------------
1. LoRA target set actually matches this checkpoint.
   config.use_mla is False, so attention is plain q/k/v/o_proj. v1/v2 targeted
   q_a_proj / q_b_proj / kv_a_proj_with_mqa / kv_b_proj -- MLA names that match
   nothing here -- and never targeted k_proj or v_proj. Inspecting v1's saved
   adapter: 4224 of its 4290 tensors sit on the 64 routed experts, only 24 on
   attention (q_proj + o_proj). Each routed expert sees ~6/64 of tokens, so 98%
   of the 76.5M "trainable" params were learning on ~9% of the gradient signal.
   v3 puts full rank on the always-on path (q,k,v,o + shared_experts + the dense
   layer-0 MLP) and a smaller rank on the routed experts (--expert-rank, 0 to
   drop them entirely).

2. Over-long targets are SKIPPED, not truncated.
   v1 cut the target at --max-target-len and appended EOS at the cut, i.e.
   taught "emit EOS in the middle of a document" on 4.3% of pages. v3 refuses
   the sample instead and reports how many it dropped.

3. Loss is computed only over the target span.
   Mathematically identical to the checkpoint's built-in labels path (same
   shift, same mean over supervised tokens), but lm_head runs on ~530 positions
   instead of ~1450, so the two float32 [1, T, 129280] logit tensors shrink ~3x.
   That is what makes a 2048-token target cap fit on 12 GB.

4. Model selection on decoded CER, not val loss.
   v2 improved val loss over v1 (0.2158 -> lower) yet its recorded test CER got
   WORSE (0.4071 -> 0.4258). Teacher-forced loss does not see decode-time
   failure modes, so v3 periodically greedy-decodes --metric-eval-pages val
   pages with the exact deployment decode settings and keeps best_cer/.

5. Crop budget matches inference (max_num=32, dynamic_preprocess's own default).

6. Warmup -> cosine LR (as v2), plus grad-norm logging and a proper
   consecutive-failure abort.

The decode-side fixes (ring KV cache, max_new_tokens, repairing formatter) live
in common_v3.py / format_v3.py and are used by both the in-training metric eval
and gen_baidu_v3.py, so training selection and deployment agree.
"""
import os, sys, json, time, math, argparse, random, shutil, statistics
from pathlib import Path

V3 = Path(__file__).resolve().parent
sys.path.insert(0, str(V3))

import torch
from common_v3 import (load_base, helpers, build_train_example, build_inputs, generate_latex,
                       forward_loss, read_jsonl, target_of, causal_lm,
                       ATTN_TARGETS, MLP_TARGETS, ROUTED_EXPERT_RE,
                       DATA_DIR, IMAGE_DIR, write_json_atomic)
from format_v3 import ft_format_v3
import metrics_v3

# PEFT matches rank/alpha pattern keys as re.match(rf".*\.{key}$", module_name)
EXPERT_PATTERN = r"mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)"


def write_metrics(path, **kw):
    try:
        cur = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
        cur.update(kw)
        cur["updated_at"] = time.time()
        write_json_atomic(path, cur)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# power-fail-safe checkpointing
#
# A save is two things: several hundred MB of adapter_model.safetensors, and a
# torch.save() of optimizer/scheduler state. Neither is atomic on its own -- a
# power cut mid-write leaves a truncated file. If that partial write landed in
# the ONE directory resume reads from, resume loads garbage or crashes.
#
# So we never overwrite the checkpoint resume trusts. Two slots (checkpoint_a /
# checkpoint_b) alternate; each save writes into the slot NOT currently
# pointed to, and only after every file in it is fully written do we update
# checkpoint_ptr.json (a single small file, replaced with os.replace() which is
# atomic on NTFS) to point at it. A crash at any point before that final
# pointer swap leaves the previous slot -- which was already known-good -- as
# the one resume will use. Worst case lost work is bounded by --save-every-s,
# not by how long a single safetensors write takes.
# --------------------------------------------------------------------------- #
def _other_slot(prev):
    return "checkpoint_b" if prev == "checkpoint_a" else "checkpoint_a"


def _resume_state_readable(slot_dir):
    """Actually try to parse resume_state.pt, not just check it exists.

    Found in production: a slot whose resume_state.pt existed, matched the
    known-good file's size AND sha256 byte-for-byte reproducibly, yet
    torch.load() raised 'failed finding central directory' every time -- a
    write that reported success but produced a file resume can't use. Existence
    checks alone would trust it and crash on the NEXT resume, possibly hours
    later. Actually loading it here is the only real verification.
    """
    try:
        torch.load(slot_dir / "resume_state.pt", map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def read_ckpt_ptr(out):
    p = out / "checkpoint_ptr.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        slot = out / d["slot"]
        if ((slot / "adapter_config.json").exists() and (slot / "resume_state.pt").exists()
                and _resume_state_readable(slot)):
            return d
    except Exception:
        pass
    return None  # missing/corrupt pointer or slot -> treat as no checkpoint, never crash on it


def save_checkpoint(model, opt, sched, out, step, best_val, best_cer, hist, prev_slot):
    target_name = _other_slot(prev_slot)
    target = out / target_name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)   # stale write from a prior crash, if any
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    torch.save({"step": step, "best_val": best_val, "best_cer": best_cer, "history": hist,
                "opt": opt.state_dict(), "sched": sched.state_dict()}, target / "resume_state.pt")
    # Read back what was just written before trusting it -- catches a write that
    # "succeeds" (no exception, correct file size) but produces an unparseable
    # file, which happened in production and would otherwise only surface on
    # the next resume, possibly hours later with no earlier checkpoint at hand.
    if not _resume_state_readable(target):
        raise RuntimeError(f"just-written {target/'resume_state.pt'} failed to read back -- "
                           f"NOT swinging the pointer, previous checkpoint stays active")
    # Only now does this slot become the one resume will trust.
    write_json_atomic(out / "checkpoint_ptr.json",
                      {"slot": target_name, "step": step, "saved_at": time.time()})
    return target_name


def atomic_save_pretrained(model, path: Path):
    """Same double-buffer idea for best_loss/best_cer/final: write to a sibling
    temp dir, then swap it in with os.replace (atomic for a single rename)
    instead of writing straight into `path`, so a crash mid-save never leaves a
    half-written directory sitting where gen_baidu_v3.py looks for an adapter."""
    tmp = path.with_name(path.name + "_tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp))
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def build_lora(args):
    from peft import LoraConfig
    cfg = dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
               target_modules=ATTN_TARGETS + MLP_TARGETS, bias="none", task_type="CAUSAL_LM",
               use_rslora=args.rslora)
    if args.expert_rank <= 0:
        cfg["exclude_modules"] = [EXPERT_PATTERN]
    elif args.expert_rank != args.lora_r:
        cfg["rank_pattern"] = {EXPERT_PATTERN: args.expert_rank}
        # keep the same alpha/r scaling on the experts as everywhere else
        cfg["alpha_pattern"] = {EXPERT_PATTERN: int(args.lora_alpha * args.expert_rank / args.lora_r)}
    return LoraConfig(**cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default=str(DATA_DIR / "train.jsonl"))
    ap.add_argument("--val-file", default=str(DATA_DIR / "val.jsonl"))
    ap.add_argument("--image-base", default=str(DATA_DIR))
    ap.add_argument("--output-dir", default=str(V3 / "out" / "baidu-ocr-math-v3"))
    ap.add_argument("--prompt", default="<image>Convert the handwriting to a complete LaTeX document.")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5, help="cosine floor (v1/v2 decayed to 0)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--expert-rank", type=int, default=8,
                    help="LoRA rank on the 64 routed MoE experts (0 = do not adapt them)")
    ap.add_argument("--rslora", action="store_true", default=True)
    ap.add_argument("--no-rslora", dest="rslora", action="store_false")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--max-target-len", type=int, default=2048,
                    help="targets longer than this are SKIPPED (never truncated); train max is 1919")
    ap.add_argument("--max-crops", type=int, default=32, help="matches dynamic_preprocess's inference default")
    ap.add_argument("--base-size", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--warmup-steps", type=int, default=0, help="0 = auto = 5%% of total steps")
    ap.add_argument("--save-steps", type=int, default=100, help="checkpoint at least this often")
    ap.add_argument("--save-every-s", type=int, default=600,
                    help="ALSO checkpoint every N wall-clock seconds regardless of --save-steps, "
                         "so a power cut loses at most this much work even if step time varies")
    ap.add_argument("--eval-steps", type=int, default=200, help="teacher-forced val loss")
    ap.add_argument("--eval-samples", type=int, default=80)
    ap.add_argument("--metric-eval-steps", type=int, default=400, help="0 = off; greedy-decode CER eval")
    ap.add_argument("--metric-eval-pages", type=int, default=24)
    ap.add_argument("--metric-max-new-tokens", type=int, default=2048)
    ap.add_argument("--metric-page-timeout", type=float, default=120.0)
    ap.add_argument("--early-stopping-patience", type=int, default=0,
                    help="consecutive CER evals without improvement before stopping (0=off)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--simple-loss", action="store_true", help="use the checkpoint's built-in labels path")
    ap.add_argument("--resume", action="store_true",
                    help="no-op kept for compatibility -- resume is now AUTOMATIC whenever a valid "
                         "checkpoint is found under --output-dir (see --fresh)")
    ap.add_argument("--reset-best-cer", action="store_true",
                    help="forget the resumed best_cer number (use when the selection metric's "
                         "definition changed, so old and new values are not comparable)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint under --output-dir and start over "
                         "(default: auto-resume if a valid one exists -- this is what makes a "
                         "power-cut recoverable by just rerunning the same command)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="build 1 example, 1 forward, 1 greedy decode, print + exit")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "training_metrics.json"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    write_metrics(metrics_path, status="loading", started_at=time.time(), history=[],
                  config={k: v for k, v in vars(args).items()})

    print(f"metrics backend: {metrics_v3.reason()}", flush=True)
    print("loading model...", flush=True)
    model, tok = load_base(args.qlora)
    M = helpers(model)
    base = Path(args.image_base)

    def to_example(row):
        return build_train_example(M, tok, base / row["image"], args.prompt, target_of(row),
                                   args.base_size, args.image_size, args.max_crops,
                                   args.max_target_len)

    # ---- smoke: one example, one forward, one decode -------------------------
    if args.smoke:
        model.to(dev).train()
        row = read_jsonl(args.train_file)[0]
        ex = to_example(row)
        if ex is None:
            sys.exit("smoke: first row's target exceeds --max-target-len")
        print(f"input_ids={tuple(ex['input_ids'].shape)} image_positions={int(ex['seq_mask'].sum())} "
              f"supervised={int((ex['labels'] != -100).sum())} crops={tuple(ex['images_crop'].shape)} "
              f"spatial={ex['spatial'].tolist()}", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = forward_loss(model, ex, dev, args.label_smoothing, args.simple_loss)
            ref = forward_loss(model, ex, dev, 0.0, True)
        print(f"SMOKE loss(sliced)={loss.item():.4f}  loss(builtin)={ref.item():.4f}  "
              f"(should match to ~1e-3)", flush=True)
        model.eval()
        pex = build_inputs(M, tok, base / row["image"], args.prompt,
                           args.base_size, args.image_size, args.max_crops)
        t0 = time.time()
        txt = generate_latex(model, tok, pex, dev, M=M, max_new_tokens=256,
                             ring=False, max_time=90)
        print(f"SMOKE decode ({time.time()-t0:.0f}s, ring OFF), first 300 chars:\n{txt[:300]}", flush=True)
        write_metrics(metrics_path, status="smoke_ok", smoke_loss=round(loss.item(), 4))
        return

    # ---- LoRA ----------------------------------------------------------------
    from peft import get_peft_model, PeftModel
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    ptr = None if args.fresh else read_ckpt_ptr(out)
    resuming = ptr is not None
    if resuming:
        ckpt = out / ptr["slot"]
        model = PeftModel.from_pretrained(model, str(ckpt), is_trainable=True)
        print(f"RESUMING adapter from {ckpt} (step {ptr['step']}, "
              f"saved {time.time() - ptr['saved_at']:.0f}s ago -- this is what recovers a power cut)",
              flush=True)
    else:
        ckpt = None
        model = get_peft_model(model, build_lora(args))
    model.to(dev)
    # Gradient checkpointing stays OFF: it needs enable_input_require_grads(),
    # which makes inputs_embeds a grad-requiring leaf, and the checkpoint's
    # forward splices vision features with an in-place masked_scatter_ on a view
    # of it -- autograd forbids that. bf16 + LoRA + batch 1 fits 12 GB.
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_attn = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and ".self_attn." in n)
    n_expert = sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and ".experts." in n and ".shared_experts." not in n)
    print(f"trainable: {trainable/1e6:.2f}M  (attention {n_attn/1e6:.2f}M, "
          f"routed-experts {n_expert/1e6:.2f}M, dense/shared {(trainable-n_attn-n_expert)/1e6:.2f}M)",
          flush=True)

    # ---- data ---------------------------------------------------------------
    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
    random.seed(args.seed)
    steps_per_epoch = max(1, len(train_rows) // args.grad_accum)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.999))

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warm_n = args.warmup_steps if args.warmup_steps > 0 else max(20, int(0.05 * total_steps))
    warm_n = min(warm_n, max(1, total_steps - 1))
    sched = SequentialLR(
        opt,
        schedulers=[LinearLR(opt, start_factor=0.05, end_factor=1.0, total_iters=warm_n),
                    CosineAnnealingLR(opt, T_max=max(1, total_steps - warm_n), eta_min=args.min_lr)],
        milestones=[warm_n])
    print(f"LR: {warm_n} warmup -> cosine to {args.min_lr:g} over {total_steps - warm_n}", flush=True)

    start_step, best_val, best_cer, hist = 0, None, None, []
    cur_slot = None if not resuming else ptr["slot"]     # tracks which slot save_checkpoint should NOT overwrite
    if resuming:
        rs = ckpt / "resume_state.pt"
        try:
            st = torch.load(rs, map_location=dev, weights_only=False)
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            start_step = int(st.get("step", 0))
            best_val, best_cer = st.get("best_val"), st.get("best_cer")
            hist = st.get("history", [])
            print(f"RESUMED optimizer at step {start_step}", flush=True)
            if args.reset_best_cer:
                print(f"  --reset-best-cer: discarding resumed best_cer={best_cer} "
                      f"(metric definition changed)", flush=True)
                best_cer = None
        except Exception as e:
            print(f"could not load resume_state ({e}) - continuing with restored weights only", flush=True)

    # ---- evaluation ----------------------------------------------------------
    loss_val_rows = val_rows[: args.eval_samples]
    metric_val_rows = val_rows[: args.metric_eval_pages]

    @torch.no_grad()
    def eval_loss():
        model.eval()
        losses, skipped = [], 0
        for row in loss_val_rows:
            ex = to_example(row)
            if ex is None:
                skipped += 1
                continue
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses.append(forward_loss(model, ex, dev, 0.0, args.simple_loss).item())
            except Exception as e:
                skipped += 1
                print(f"    eval-loss skip {row['id']}: {type(e).__name__}", flush=True)
        model.train()
        if skipped:
            print(f"    eval-loss used {len(losses)}/{len(loss_val_rows)} pages", flush=True)
        return (sum(losses) / len(losses)) if losses else None

    @torch.no_grad()
    def eval_cer():
        """Greedy-decode with the exact deployment settings and score. This is the
        signal v1/v2 never had -- teacher-forced loss cannot see a decode loop.

        Reported three ways, because a 24-page MEAN is fragile: one page that
        enters a repetition loop emits up to the full token budget and scores
        CER >> 1 (CER is unbounded above -- it is edit distance over REFERENCE
        length, so a 4x-too-long output scores ~4.0), single-handedly moving the
        mean by ~0.17. That is not a model regression, it is one bad page.
          cer         uncapped mean -- comparable to the 700-page benchmark number
          cer_capped  mean of per-page min(cer, 1.0) -- SELECTION signal; bounds
                      any single blown-up page to 1/n influence
          cer_median  robust centre, ignores blowups entirely
        Per-page rows are dumped so a bad eval can actually be diagnosed instead
        of guessed at.
        """
        model.eval()
        rows_out = []
        for row in metric_val_rows:
            try:
                ex = build_inputs(M, tok, base / row["image"], args.prompt,
                                  args.base_size, args.image_size, args.max_crops)
                raw = generate_latex(model, tok, ex, dev, M=M,
                                     max_new_tokens=args.metric_max_new_tokens,
                                     ring=False, max_time=args.metric_page_timeout)
                hyp = ft_format_v3(raw)
                r = metrics_v3.score(target_of(row), hyp)
                r["complete"] = "\\end{document}" in raw
                r["id"] = row.get("id")
                r["raw_chars"] = len(raw)
                rows_out.append(r)
            except Exception as e:
                print(f"    eval-cer skip {row.get('id')}: {type(e).__name__}", flush=True)
            finally:
                torch.cuda.empty_cache()
        model.train()
        if not rows_out:
            return None
        n = len(rows_out)
        cers = [r["cer"] for r in rows_out]
        capped = [min(1.0, c) for c in cers]
        worst = sorted(rows_out, key=lambda r: -r["cer"])[:3]
        try:
            (out / f"eval_pages_step{step}.json").write_text(
                json.dumps({"step": step, "rows": rows_out}, indent=1), encoding="utf-8")
        except Exception:
            pass
        return {"cer": sum(cers) / n,
                "cer_capped": sum(capped) / n,
                "cer_median": statistics.median(cers),
                "blown_up": sum(1 for c in cers if c > 1.0),
                "worst": [(r.get("id"), round(r["cer"], 2), r.get("len_rate")) for r in worst],
                "ncer": sum(r["ncer"] for r in rows_out) / n,
                "struct_pct": 100.0 * sum(bool(r["struct"]) for r in rows_out) / n,
                "complete_pct": 100.0 * sum(r["complete"] for r in rows_out) / n,
                "pages": n}

    write_metrics(metrics_path, status="running", step=start_step, max_steps=total_steps,
                  epochs=args.epochs, trainable_m=round(trainable / 1e6, 2), history=hist)
    print(f"training: {len(train_rows)} samples, {total_steps} steps "
          f"(accum {args.grad_accum}, ~{args.epochs} epochs)", flush=True)

    # ---- loop ----------------------------------------------------------------
    step, t0, micro = start_step, time.time(), 0
    fails, no_improve, skipped_long, aborted = 0, 0, 0, False
    opt.zero_grad(set_to_none=True)
    stop = step >= total_steps
    last_ckpt_time = time.time()

    def checkpoint_now(reason, required=False):
        """required=True (used on the pre-abort/end-of-run paths) lets a failure
        propagate -- those are last chances to save. The periodic path instead
        logs and returns without moving last_ckpt_time, so due_by_time re-fires
        and it retries at the next opportunity instead of losing the run over
        one flaky write; the pointer is never at risk either way since
        save_checkpoint only swings it after a successful read-back."""
        nonlocal cur_slot, last_ckpt_time
        t1 = time.time()
        try:
            cur_slot = save_checkpoint(model, opt, sched, out, step, best_val, best_cer, hist, cur_slot)
        except Exception as e:
            print(f"  checkpoint FAILED ({reason}): {type(e).__name__}: {e} "
                  f"-- previous checkpoint at {cur_slot} remains active", flush=True)
            if required:
                raise
            return
        last_ckpt_time = time.time()
        print(f"  checkpoint -> {cur_slot} at step {step} ({reason}, {last_ckpt_time - t1:.0f}s)", flush=True)

    for epoch in range(math.ceil(args.epochs)):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for ri in order:
            if stop:
                break
            try:
                ex = to_example(train_rows[ri])
                if ex is None:
                    skipped_long += 1
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = forward_loss(model, ex, dev, args.label_smoothing, args.simple_loss) / args.grad_accum
                loss.backward()
                fails = 0
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                micro = 0
                fails += 1
                print("  OOM on a sample - skipped (partial accumulation dropped)", flush=True)
                if fails >= 80:
                    if step > start_step:
                        try:
                            checkpoint_now("pre-abort")
                        except Exception as ce:
                            print(f"  pre-abort checkpoint failed: {ce}", flush=True)
                    write_metrics(metrics_path, status="error",
                                  error="too many consecutive OOMs - try --qlora, --expert-rank 0 or lower --max-crops")
                    print("ABORT: 80 consecutive OOMs", flush=True)
                    aborted = stop = True
                continue
            except Exception as e:
                fails += 1
                print(f"  sample error: {type(e).__name__}: {str(e)[:100]}", flush=True)
                if fails >= 60:
                    if step > start_step:
                        try:
                            checkpoint_now("pre-abort")
                        except Exception as ce:
                            print(f"  pre-abort checkpoint failed: {ce}", flush=True)
                    write_metrics(metrics_path, status="error",
                                  error=f"systemic error: {type(e).__name__}: {str(e)[:120]}")
                    print("ABORT: 60 consecutive sample errors", flush=True)
                    aborted = stop = True
                continue

            micro += 1
            if micro < args.grad_accum:
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            micro = 0
            step += 1

            cur_loss = loss.item() * args.grad_accum
            el = time.time() - t0
            eta = el / max(1, step - start_step) * (total_steps - step)
            if step % 5 == 0 or step <= start_step + 2:
                hist = (hist + [{"step": step, "loss": round(cur_loss, 4)}])[-600:]
                print(f"  step {step}/{total_steps} loss={cur_loss:.4f} "
                      f"gnorm={float(gnorm):.2f} lr={sched.get_last_lr()[0]:.2e} "
                      f"eta={eta/60:.0f}m", flush=True)
            write_metrics(metrics_path, status="running", step=step, max_steps=total_steps,
                          loss=round(cur_loss, 4), lr=sched.get_last_lr()[0],
                          grad_norm=round(float(gnorm), 3),
                          percent=round(100 * step / total_steps, 1), eta_s=int(eta),
                          epoch=epoch, history=hist, best_val=best_val, best_cer=best_cer,
                          skipped_long=skipped_long)

            if args.eval_steps and step % args.eval_steps == 0:
                vl = eval_loss()
                if vl is not None:
                    improved = best_val is None or vl < best_val - 1e-4
                    if improved:
                        best_val = vl
                        atomic_save_pretrained(model, out / "best_loss")
                    print(f"  val_loss {vl:.4f}{' *best*' if improved else ''}", flush=True)
                    hist = (hist + [{"step": step, "val_loss": round(vl, 4)}])[-600:]
                    write_metrics(metrics_path, val_loss=round(vl, 4),
                                  best_val=round(best_val, 4), history=hist)

            if args.metric_eval_steps and step % args.metric_eval_steps == 0:
                t1 = time.time()
                mv = eval_cer()
                if mv is not None:
                    # Select on the CAPPED mean: robust to a single looping page,
                    # which the raw mean is not (see eval_cer's docstring).
                    sel = mv["cer_capped"]
                    improved = best_cer is None or sel < best_cer - 1e-4
                    if improved:
                        best_cer = sel
                        no_improve = 0
                        atomic_save_pretrained(model, out / "best_cer")
                    else:
                        no_improve += 1
                    print(f"  decode-eval CERcap={sel:.4f} (raw={mv['cer']:.4f} "
                          f"med={mv['cer_median']:.4f} blown_up={mv['blown_up']}/{mv['pages']}) "
                          f"nCER={mv['ncer']:.4f} struct={mv['struct_pct']:.0f}% "
                          f"complete={mv['complete_pct']:.0f}% ({time.time()-t1:.0f}s)"
                          f"{' *best*' if improved else f' (no improvement {no_improve})'}", flush=True)
                    print(f"    worst pages (id, cer, len_rate): {mv['worst']}", flush=True)
                    hist = (hist + [{"step": step, "eval_cer": round(mv["cer"], 4),
                                     "eval_cer_capped": round(sel, 4),
                                     "eval_cer_median": round(mv["cer_median"], 4),
                                     "eval_struct": round(mv["struct_pct"], 1)}])[-600:]
                    write_metrics(metrics_path, eval_cer=round(mv["cer"], 4),
                                  eval_cer_capped=round(sel, 4),
                                  eval_cer_median=round(mv["cer_median"], 4),
                                  eval_blown_up=mv["blown_up"], eval_worst=mv["worst"],
                                  best_cer=round(best_cer, 4), eval_struct=mv["struct_pct"],
                                  eval_complete=mv["complete_pct"], history=hist)
                    if args.early_stopping_patience and no_improve >= args.early_stopping_patience:
                        print(f"EARLY STOP: decoded CER flat for {no_improve} evals "
                              f"- best={best_cer:.4f}", flush=True)
                        stop = True

            due_by_steps = args.save_steps and step % args.save_steps == 0
            due_by_time = args.save_every_s and (time.time() - last_ckpt_time) >= args.save_every_s
            if due_by_steps or due_by_time:
                checkpoint_now("steps" if due_by_steps else "time")

            if step >= total_steps:
                stop = True
        if stop:
            break

    if aborted:
        print("TRAINING ABORTED - status=error kept; a checkpoint was saved if any steps completed, "
              "'final' not written. Rerun the same command to resume.", flush=True)
        sys.exit(1)   # non-zero so an outer retry loop (run_v3.ps1 train) knows to relaunch
    else:
        if step > start_step:
            checkpoint_now("end-of-run")   # so a crash between here and the atomic 'final' swap is still recoverable
        atomic_save_pretrained(model, out / "final")
        write_metrics(metrics_path, status="done", step=step, max_steps=total_steps,
                      best_val=best_val, best_cer=best_cer, skipped_long=skipped_long,
                      finished_at=time.time())
        print(f"TRAINING DONE  best_val={best_val}  best_cer={best_cer}  "
              f"skipped_long_targets={skipped_long}", flush=True)


if __name__ == "__main__":
    main()
