# Sample pages

Two pages from the held-out validation split, included to show the pipeline's
input/output at a few stages without shipping the full (confidential) dataset.

## PII safety note

The full dataset carries personally identifiable information (student names,
roll numbers) that must never be published. Every image here was **manually
inspected by a human reviewer** before being added to this repo — not just
run through the automated redaction step, which is known to miss non-standard
page layouts (see `pipeline/anonymize.py`'s top-%-whiteout method, and the
note in the main README's Limitations section).

Original filenames from the raw dataset encode source-batch identifiers and
were **not** used here — files are renamed generically (`sample_A`, `sample_B`)
so no batch/source metadata is exposed alongside the images.

## What's included, per sample

| File | Pipeline stage | Contents |
|---|---|---|
| `1_cropped.jpg` | Stage 1 output | After PII crop/redaction, before filtering |
| `5_prepared.png` | Stage 5 output | Deskewed, converted, resized — this is what the model actually sees |
| `6_annotation_raw.tex` | Stage 6 output | Teacher-VLM's raw annotation, before quality review |
| `7_validated.tex` | Stage 7 output | Final annotation, verified to compile with `pdflatex` |

For both samples here, the raw and validated `.tex` are identical — quality
review found nothing to fix on these two pages. That won't be true for every
page in the full corpus (14,772 → 14,560 pages survive Stage 7→8, see the main
README's pipeline table).
