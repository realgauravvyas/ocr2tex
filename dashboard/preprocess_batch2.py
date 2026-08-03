"""Standalone preprocessing for the scans2- batch (crop -> filter -> deskew -> convert).

Runs independently of the dashboard so batch-2 CPU work proceeds while batch-1
annotation is still running. Writes into the same workspace folders with
skip-if-exists semantics, so the dashboard pipeline picks up at the prepare
stage afterwards.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, r"D:\ocr2tex\dashboard")
from app import crop_single, filter_single, deskew_single, convert_single

RAW = Path(r"D:\ocr2tex\Data\raw data")
WORK = Path(r"D:\ocr2tex\workspace")
CROP = WORK / "1_cropped"
KEEP = WORK / "2_filtered"
REJ = WORK / "2_filtered_rejected"
DESK = WORK / "3_deskewed"
CONV = WORK / "4_converted_png"
WORKERS = 8
CROP_PCT = 10
DARK_THRESH = 210
MAX_DARK_PCT = 0.5
PREFIX = "scans2-"


def run_stage(name, tasks, worker):
    if not tasks:
        print(f"{name}: nothing to do", flush=True)
        return
    t0 = time.time()
    counts = {}
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            status, src, err = fut.result()[:3]
            counts[status] = counts.get(status, 0) + 1
            if status == "fail":
                print(f"{name} FAIL: {Path(src).name} - {err}", flush=True)
            done += 1
            if done % 2000 == 0:
                print(f"{name}: {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"{name} done: {counts} in {time.time()-t0:.0f}s", flush=True)


def main():
    for p in (CROP, KEEP, REJ, DESK, CONV):
        p.mkdir(parents=True, exist_ok=True)

    src_files = sorted(RAW.glob(PREFIX + "*.jpg"))
    print(f"batch2 source files: {len(src_files)}", flush=True)

    tasks = [(str(f), str(CROP / f.name), CROP_PCT) for f in src_files if not (CROP / f.name).exists()]
    run_stage("crop", tasks, crop_single)

    cropped = sorted(CROP.glob(PREFIX + "*.jpg"))
    tasks = [
        (str(f), str(KEEP), str(REJ), DARK_THRESH, MAX_DARK_PCT)
        for f in cropped
        if not (KEEP / f.name).exists() and not (REJ / f.name).exists()
    ]
    run_stage("filter", tasks, filter_single)

    kept = sorted(KEEP.glob(PREFIX + "*.jpg"))
    tasks = [(str(f), str(DESK / f.name)) for f in kept if not (DESK / f.name).exists()]
    run_stage("deskew", tasks, deskew_single)

    desk = sorted(DESK.glob(PREFIX + "*.jpg"))
    tasks = [(str(f), str(CONV / (f.stem + ".png"))) for f in desk if not (CONV / (f.stem + ".png")).exists()]
    run_stage("convert", tasks, convert_single)

    print("PREPROCESS_BATCH2_DONE", flush=True)
    print(
        f"kept={len(list(KEEP.glob(PREFIX + '*.jpg')))} "
        f"rejected={len(list(REJ.glob(PREFIX + '*.jpg')))} "
        f"converted={len(list(CONV.glob(PREFIX + '*.png')))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
