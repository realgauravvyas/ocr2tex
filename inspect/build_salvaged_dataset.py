"""Package salvaged (temp-0.45) pages as a SEPARATE dataset for future use.

- Reads workspace/salvaged_pages.json (which pages came from the salvage pass)
- Takes the validated LaTeX from 7_validated (compile-checked, auto-fixed)
- Copies images and writes salvaged_dataset/salvaged.jsonl in the same training
  format as the main dataset, with an extra "verified" flag from the
  self-consistency report (two independent samples agreed within threshold)
- Writes workspace/excluded_pages.json so the main build/split never includes
  these pages
"""

import json
import shutil
from datetime import date
from pathlib import Path

WORK = Path(r"D:\ocr2tex\workspace")
OUT = WORK / "salvaged_dataset"

TRAINING_SYSTEM_PROMPT = None
TRAINING_USER_PROMPT = None


def load_prompts():
    global TRAINING_SYSTEM_PROMPT, TRAINING_USER_PROMPT
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import app
    TRAINING_SYSTEM_PROMPT = app.TRAINING_SYSTEM_PROMPT
    TRAINING_USER_PROMPT = app.TRAINING_USER_PROMPT


def main():
    load_prompts()
    excl_path = WORK / "excluded_pages.json"
    if excl_path.exists():
        salvaged = json.loads(excl_path.read_text(encoding="utf-8"))["pages"]
    else:
        salvaged = json.loads((WORK / "salvaged_pages.json").read_text(encoding="utf-8"))["pages"]
    verify = {}
    vr_path = WORK / "verify_report.json"
    if vr_path.exists():
        verify = json.loads(vr_path.read_text(encoding="utf-8")).get("details", {})

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    validated_dir = WORK / "7_validated"
    prepared_dir = WORK / "5_prepared"

    samples = []
    no_validated = 0
    for stem in salvaged:
        tex_file = validated_dir / f"{stem}.tex"
        img_file = prepared_dir / f"{stem}.png"
        if not tex_file.exists() or not img_file.exists():
            no_validated += 1
            continue
        latex = tex_file.read_text(encoding="utf-8").strip()
        v = verify.get(stem, {})
        shutil.copy2(str(img_file), str(img_dir / f"{stem}.png"))
        samples.append({
            "id": stem,
            "image": f"images/{stem}.png",
            "verified": v.get("verdict") == "keep",
            "self_consistency_cer": v.get("cer"),
            "conversations": [
                {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                {"role": "user", "content": TRAINING_USER_PROMPT},
                {"role": "assistant", "content": latex},
            ],
        })

    with open(OUT / "salvaged.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if vr_path.exists():
        shutil.copy2(str(vr_path), str(OUT / "verify_report.json"))

    n_verified = sum(1 for s in samples if s["verified"])
    (OUT / "README.md").write_text(f"""# Salvaged dataset ({date.today()})

Pages that failed annotation at temperature 0.1 (deterministic reasoning loops)
and were recovered at temperature 0.45. EXCLUDED from the main training dataset
(workspace/9_split) per the project decision to keep the main set temp-0.1 only.

- {len(samples)} samples, all pdflatex-validated
- {n_verified} passed self-consistency verification ("verified": true): a second
  independent annotation at the same temperature agreed within 10% character
  error rate - strong evidence the transcription is anchored to the page
- For future training, prefer filtering to "verified": true

Format matches the main dataset (same system/user prompts).
""", encoding="utf-8")

    # durable exclusion from all future main-dataset builds
    (WORK / "excluded_pages.json").write_text(
        json.dumps({"reason": "salvaged at temperature 0.45 - kept separate", "pages": salvaged}, indent=2),
        encoding="utf-8",
    )

    print(f"salvaged dataset: {len(samples)} samples ({n_verified} verified) -> {OUT}")
    print(f"excluded from main dataset: {len(salvaged)} pages ({no_validated} had no validated tex)")


if __name__ == "__main__":
    main()
