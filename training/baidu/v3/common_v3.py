"""Shared plumbing for the BaiduOCR v3 experiment.

ISOLATION CONTRACT — nothing here writes outside D:\\Claude Code\\BaiduOCR-v3:
  * the dataset (D:\\ocr2tex\\workspace\\9_split) is opened read-only
  * the v1 adapter (D:\\Claude Code\\BaiduOCR\\finetune\\...) is opened read-only
  * D:\\ocr2tex\\output\\* (bench_outputs, benchmark_results.json) is NEVER touched
    -- v3 writes its own bench dir and its own results file under out/ and bench/

WHY v3 EXISTS (measured on the 700-page test set, v1 adapter):
  1. Unlimited-OCR's attention class is SlidingWindowLlamaAttention (config
     use_mla=False) and infer() arms a 128-slot RING KV cache before generate()
     (config._ring_window = sliding_window_size = 128). Training runs with
     use_cache=False, i.e. FULL causal attention. So the model is trained to see
     all previously-emitted tokens but at decode time only ever sees the prompt
     plus the last 128 generated tokens, written into recycled slots that still
     carry their original RoPE phase. Targets average 532 tokens. That is the
     train/inference mismatch behind the degenerate repetition:
       124/700 outputs never emitted \\end{document}; 103 of those 124 contain an
       8-gram repeated >=3 times, vs a median repeat of 2 on the clean pages.
     v3 disables the ring at decode time so inference matches training.
  2. LORA_TARGETS in v1/v2 listed MLA projection names (q_a_proj, q_b_proj,
     kv_a_proj_with_mqa, kv_b_proj) that match NOTHING in this checkpoint, and
     omitted k_proj/v_proj which do exist. Result: of 76.53M trainable params,
     4224 of the 4290 LoRA tensors sit on the 64 routed experts (each seeing
     ~6/64 of tokens) and attention got only q_proj + o_proj.
  3. max_length=2048 at inference is a TOTAL cap; the image prompt alone is
     ~903 tokens (2x3 crops), leaving ~1145 for the answer. v3 uses
     max_new_tokens so the answer budget no longer depends on page geometry.
  4. Training truncated targets at --max-target-len and appended EOS at the cut,
     teaching the model to stop mid-document. v3 SKIPS over-long targets instead.
  5. dynamic_preprocess default max_num=32 at inference vs max_num=16 in v1
     training. v3 uses 32 on both sides.
"""
import os, sys, json, math, re
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from PIL import ImageOps
import torch

MODEL_NAME = "baidu/Unlimited-OCR"
IMAGE_TOKEN_ID = 128815

# Verified against the checkpoint's own module tree (use_mla=False => plain
# q/k/v/o attention; MoE mlp with 64 routed experts + 2 shared + a dense layer 0).
ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
# regex over module names, used for PEFT rank_pattern/alpha_pattern
ROUTED_EXPERT_RE = r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)"

DATA_DIR = Path(r"D:\ocr2tex\workspace\9_split")          # READ-ONLY
IMAGE_DIR = DATA_DIR / "images"                            # READ-ONLY
V3_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def patch_tf():
    """transformers >= 4.50 moved is_torch_fx_available; the remote code imports
    it from the old location."""
    try:
        from transformers.utils.import_utils import is_torch_fx_available  # noqa
    except ImportError:
        import transformers.utils.import_utils as _i, transformers.utils as _t
        _i.is_torch_fx_available = _t.is_torch_fx_available = lambda: False


def load_base(qlora=False, dtype=torch.bfloat16):
    patch_tf()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    kw = dict(trust_remote_code=True, use_safetensors=True, torch_dtype=dtype)
    if qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModel.from_pretrained(MODEL_NAME, **kw)
    return model, tok


def helpers(model):
    """The checkpoint's own preprocessing functions, so our tensors are built by
    exactly the code infer() uses."""
    mod = sys.modules[type(model).__module__]
    return dict(fmt=mod.format_messages, load_imgs=mod.load_pil_images,
                t_enc=mod.text_encode, dyn=mod.dynamic_preprocess,
                Tfm=mod.BasicImageTransform,
                NgramProc=getattr(mod, "SlidingWindowNoRepeatNgramProcessor", None))


def causal_lm(model):
    """Unwrap a PeftModel to the UnlimitedOCRForCausalLM (LoRA layers stay
    injected in-place, so this still runs the adapter)."""
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


# --------------------------------------------------------------------------- #
# BUG 1 — the 128-slot ring KV cache
# --------------------------------------------------------------------------- #
def set_ring(model, enabled: bool, window: int = 128):
    """Arm or disarm the ring KV cache for generation.

    SlidingWindowLlamaAttention reads `config._ring_window`. When it is not None
    and a KV cache exists, every generated token past the warm-up overwrites a
    slot in a `window`-sized ring, so the model can only attend to the prompt +
    the last `window` generated tokens. Training never has a cache, so it always
    sees full attention -- disarming this is what makes decode match training.

    Also pins config.sliding_window/sliding_window_size to None so that infer(),
    if it is ever used, recomputes _orig_sw as None and cannot re-arm the ring
    behind our back.
    """
    cfg = causal_lm(model).config
    if enabled:
        cfg.sliding_window_size = window
        cfg.sliding_window = None      # infer() also nulls this during generate
        cfg._ring_window = window
    else:
        cfg.sliding_window_size = None
        cfg.sliding_window = None
        cfg._ring_window = None
    return cfg


# --------------------------------------------------------------------------- #
# input construction (byte-identical to infer(), crop_mode=True)
# --------------------------------------------------------------------------- #
def build_inputs(M, tok, image_path, prompt, base_size=1024, image_size=640, max_crops=32):
    """Replicates infer()'s crop_mode input construction. Prompt only, no target.

    max_crops=32 matches dynamic_preprocess's own default, which is what infer()
    uses -- v1/v2 trained at 16 and decoded at 32.
    """
    conv = [{"role": "<|User|>", "content": prompt, "images": [str(image_path)]},
            {"role": "<|Assistant|>", "content": ""}]
    prompt_str = M["fmt"](conversations=conv, sft_format="plain", system_prompt="")
    images = M["load_imgs"](conv)
    tfm = M["Tfm"](mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    patch_size, ds = 16, 4

    splits = prompt_str.split("<image>")
    imgs_global, imgs_crop, seq_mask, ids, spatial = [], [], [], [], []
    for text_sep, image in zip(splits, images):
        t = M["t_enc"](tok, text_sep, bos=False, eos=False)
        ids += t
        seq_mask += [False] * len(t)

        if image.size[0] <= 640 and image.size[1] <= 640:
            crop_ratio, crop_raw = [1, 1], []
        else:
            crop_raw, crop_ratio = M["dyn"](image, max_num=max_crops)

        gv = ImageOps.pad(image, (base_size, base_size),
                          color=tuple(int(x * 255) for x in tfm.mean))
        imgs_global.append(tfm(gv).to(torch.bfloat16))
        wc, hc = crop_ratio
        spatial.append([wc, hc])
        if wc > 1 or hc > 1:
            for ci in crop_raw:
                imgs_crop.append(tfm(ci).to(torch.bfloat16))

        nq = math.ceil((image_size // patch_size) / ds)
        nqb = math.ceil((base_size // patch_size) / ds)
        timg = ([IMAGE_TOKEN_ID] * nqb + [IMAGE_TOKEN_ID]) * nqb + [IMAGE_TOKEN_ID]
        if wc > 1 or hc > 1:
            timg += ([IMAGE_TOKEN_ID] * (nq * wc) + [IMAGE_TOKEN_ID]) * (nq * hc)
        ids += timg
        seq_mask += [True] * len(timg)

    t = M["t_enc"](tok, splits[-1], bos=False, eos=False)
    ids += t
    seq_mask += [False] * len(t)
    ids = [0] + ids                      # bos
    seq_mask = [False] + seq_mask

    return dict(
        input_ids=torch.LongTensor(ids),
        seq_mask=torch.tensor(seq_mask, dtype=torch.bool),
        images_ori=torch.stack(imgs_global, 0),
        images_crop=(torch.stack(imgs_crop, 0) if imgs_crop
                     else torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16)),
        spatial=torch.tensor(spatial, dtype=torch.long),
        prompt_len=len(ids),
        crop_ratio=tuple(spatial[0]) if spatial else (1, 1),
    )


def build_train_example(M, tok, image_path, prompt, target, base_size=1024,
                        image_size=640, max_crops=32, max_target_len=2048):
    """Prompt tensors + target tokens + labels (-100 on prompt/image positions).

    BUG 4 fix: an over-long target is SKIPPED (returns None), never truncated.
    v1/v2 cut the target at the cap and then appended EOS, which is a direct
    lesson in 'stop in the middle of a document'.
    """
    ex = build_inputs(M, tok, image_path, prompt, base_size, image_size, max_crops)
    tgt = M["t_enc"](tok, target, bos=False, eos=False)
    if len(tgt) + 1 > max_target_len:
        return None
    tgt = tgt + [tok.eos_token_id]                      # eos_token_id == 1

    p = ex["prompt_len"]
    ex["input_ids"] = torch.cat([ex["input_ids"], torch.LongTensor(tgt)])
    ex["seq_mask"] = torch.cat([ex["seq_mask"], torch.zeros(len(tgt), dtype=torch.bool)])
    ex["labels"] = torch.LongTensor([-100] * p + tgt)
    ex["target_len"] = len(tgt)
    return ex


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def forward_loss(model, ex, dev, label_smoothing=0.0, simple=False):
    """Cross-entropy on the target span only.

    `simple=True` uses the checkpoint's built-in labels path. The default path is
    numerically identical (same shift, same mean over supervised tokens) but runs
    lm_head over only the ~530 target positions instead of all ~1450, which
    avoids two full-sequence float32 logit tensors (129280 vocab -> ~0.75 GB
    each on a 1.5k sequence). That headroom is what lets v3 raise
    --max-target-len without OOMing a 12 GB card.
    """
    kw = dict(
        input_ids=ex["input_ids"].unsqueeze(0).to(dev),
        attention_mask=torch.ones((1, ex["input_ids"].shape[0]), dtype=torch.long, device=dev),
        images=[(ex["images_crop"].to(dev, torch.bfloat16), ex["images_ori"].to(dev, torch.bfloat16))],
        images_seq_mask=ex["seq_mask"].unsqueeze(0).to(dev),
        images_spatial_crop=ex["spatial"].to(dev),
        use_cache=False,
    )
    if simple:
        return model(labels=ex["labels"].unsqueeze(0).to(dev), **kw).loss

    clm = causal_lm(model)
    hidden = clm.model(return_dict=True, **kw)[0]              # [1, T, H]
    p = ex["prompt_len"]
    # token t is predicted from hidden[t-1]; supervised tokens start at index p
    logits = clm.lm_head(hidden[:, p - 1:-1, :]).float()       # [1, L, V]
    labels = ex["labels"][p:].to(dev)                          # [L]
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]), labels.view(-1),
        ignore_index=-100, label_smoothing=label_smoothing)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
EOS_STR = "<｜end▁of▁sentence｜>"


@torch.no_grad()
def generate_latex(model, tok, ex, dev, M=None, max_new_tokens=2048, ring=False,
                   no_repeat_ngram_size=0, ngram_window=0, max_time=None):
    """Greedy decode, mirroring infer()'s generate() call but with our fixes:
      * ring KV cache off by default (matches training-time attention)
      * max_new_tokens instead of a total max_length, so the answer budget does
        not shrink when a tall page produces more crops
    Returns the decoded continuation with the EOS marker stripped.
    """
    clm = causal_lm(model)
    set_ring(model, ring)

    ids = ex["input_ids"].unsqueeze(0).to(dev)
    kw = dict(
        input_ids=ids,
        images=[(ex["images_crop"].to(dev, torch.bfloat16), ex["images_ori"].to(dev, torch.bfloat16))],
        images_seq_mask=ex["seq_mask"].unsqueeze(0).to(dev),
        images_spatial_crop=ex["spatial"],      # infer() leaves this on CPU
        do_sample=False,
        temperature=None,
        eos_token_id=tok.eos_token_id,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    if max_time:
        kw["max_time"] = float(max_time)
    if no_repeat_ngram_size > 0:
        proc = (M or {}).get("NgramProc") if M else None
        if ngram_window > 0 and proc is not None:
            kw["logits_processor"] = [proc(no_repeat_ngram_size, ngram_window)]
        else:
            kw["no_repeat_ngram_size"] = no_repeat_ngram_size

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = clm.generate(**kw)
    text = tok.decode(out[0, ids.shape[1]:])
    if text.endswith(EOS_STR):
        text = text[:-len(EOS_STR)]
    return text.strip()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def target_of(row):
    return next((c["content"] for c in row["conversations"] if c["role"] in ("assistant", "gpt")), "")


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)
