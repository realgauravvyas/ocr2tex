# Baidu Unlimited-OCR fine-tunes — comparison baseline

LoRA fine-tunes of [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) on the same
12,575-page corpus as GLM-OCR v4.1. These exist to answer one question: *is the small,
purpose-fit GLM-OCR base actually the better starting point, or does a bigger general OCR model
win once you fine-tune it too?*

Answer: GLM-OCR wins where it counts. The Baidu fine-tunes roughly halve stock CER, but they
top out around 63–65% compile rate against GLM-OCR's 82–89%, at ~2x the latency.

**Adapters:** [ctogaurav/Unlimited-OCR-math-latex](https://huggingface.co/ctogaurav/Unlimited-OCR-math-latex)

## The three runs

| | v1 | v2 | v3 |
|---|---|---|---|
| Script | `train_unlimited_ocr.py` | `train_baidu_v2.py` | `v3/train_baidu_v3.py` |
| LoRA r / alpha | 16 / 32 | 16 / 32 | 32 / 64, rsLoRA |
| Expert ranks | inherited (r=16) | inherited (r=16) | r=8, alpha=16 |
| Trainable params | 76.53 M | 76.53 M | 44.72 M |
| Epochs / steps | 1.0 / 1,571 | 1.0 / 1,571 | 2.0 planned, **stopped at 2,400 / 3,142** |
| `max_target_len` | 1024 | 2048 | 2048 |
| Best val loss | 0.2158 | **0.2096** | 0.2213 |
| Benchmark row | "Baidu OCR FT" | "Baidu OCR FT v2" | *not in `benchmark_results.json`* |

**v2 is the one to use.** v1's `max_target_len=1024` sits below the p99 target length of 1,256
tokens, so it truncated roughly 5% of training targets mid-document. v2 fixes that and adds LR
warmup plus early stopping.

v1 still shows a *better mean CER* than v2 (0.4071 vs 0.4258) — that is the truncation flattering
it. On median CER, compile rate, structural validity and every content metric, v2 wins.

## Read this before quoting v3's numbers

v3 scores mean CER **0.3632**, which looks like it beats GLM-OCR v4.1's 0.3816. It does not, and
the two numbers should not be put in the same table.

v3 was scored through `format_v3.ft_format_v3(repair=True, trim_repeats=True)` — a formatter that
repairs malformed LaTeX and trims degenerate repeated tails before scoring. Every other number in
this repo, including all of `benchmark_results.json` and the report, used the legacy wrap-only
formatter with no such post-processing. Part of v3's gain is the model and part is the formatter,
and this project has not separated the two.

To make it comparable, either re-score v3 with the legacy formatter:

```
python v3/score_v3.py --no-repair --no-trim
```

or re-score the other systems with repair and trim enabled. Until one of those is done, v3's
numbers stand alone. Its training run is also unfinished (76.4% of planned steps), and the
published adapter is the `best_cer` checkpoint from that partial run.

## Layout

```
train_unlimited_ocr.py          v1 training
train_baidu_v2.py               v2 training
gen_baidu_for_benchmark.py      generate predictions with the STOCK model (baseline row)
gen_baidu_ft_for_benchmark.py   generate predictions with a fine-tuned adapter
gen_baidu_ft_v2.py              v2 generation (final checkpoint)
gen_baidu_ft_v2_best.py         v2 generation (best checkpoint)
run_both_benchmarks.py          run stock + fine-tuned generation back to back
resume_benchmarks.py            resume an interrupted generation run
ocr_dashboard.py                live training/generation monitor (Flask)

v3/
  train_baidu_v3.py             v3 training — rsLoRA, MoE-aware ranks, resumable
  common_v3.py                  shared model/prompt/collate logic
  gen_baidu_v3.py               v3 generation for the benchmark
  score_v3.py                   v3 scorer (--no-repair / --no-trim for legacy parity)
  format_v3.py                  output formatter: LaTeX repair + degenerate-tail trimming
  metrics_v3.py                 CER / struct / compile metric implementations
  probe_decode.py               decode-time debugging for repetition loops
  dashboard_v3.py               v3 training monitor
  run_v3.ps1                    end-to-end orchestration
  register_startup_resume.ps1   re-arm the run after a reboot (it spanned several)
```

## Notes

- All paths in these scripts are argparse defaults pointing at the original machine's layout
  (`D:\ocr2tex\workspace\9_split`). Override them; nothing is hardcoded past the defaults.
- Everything ran on one RTX 3060 12GB. `max_crops` and `max_target_len` are the two knobs that
  bound VRAM.
- The training corpus is student work and is not published. The pipeline that builds an
  equivalent corpus from your own scans is in [`../../pipeline/`](../../pipeline/).
- One config wart worth knowing if you reuse these: v1/v2 `target_modules` lists MLA names
  (`q_a_proj`, `kv_a_proj_with_mqa`, `q_b_proj`, `kv_b_proj`) that never match, because this
  decoder is full MHA. In practice only `q_proj` and `o_proj` were adapted in attention — `k_proj`
  and `v_proj` were never in the list. v3 targets `q/k/v/o_proj` properly.
