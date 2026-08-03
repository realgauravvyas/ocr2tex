"""
Step 2: Annotation Helper
Uses vision models (via OrcaRouter auto mode) to generate LaTeX annotations.

How it works:
1. Reads each image from dataset/images/
2. Sends it to a vision model (auto-routed to best available)
3. Model returns a complete LaTeX document for that page
4. Saves the LaTeX as a .tex file in dataset/annotations/

Key Features:
- RESUME support: If it stops (credit runs out, error, etc.), just run again
  with a new API key and it picks up where it left off automatically.
- Only processes images that don't already have a .tex annotation.
- Progress saved after every batch.

Modes:
  --mode dry-run   : Process N images (default 5), deletes previous dry-run files
  --mode auto      : Process all REMAINING unannotated images (resumes automatically)
  --mode resume    : Same as auto but explicitly signals "I'm continuing with new key"
  --mode review    : Open each annotation + image for manual review
  --mode status    : Show progress stats (how many done, how many remaining)
"""

import os
import json
import base64
import time
import argparse
from pathlib import Path
from openai import OpenAI


# === CONFIGURATION ===
API_BASE_URL = "https://api.orcarouter.ai/v1"
API_KEY = os.environ.get("ORCAROUTER_API_KEY")
MODEL = "orcarouter/auto"  # Auto-routes to best vision model


def get_client(api_key: str = None, base_url: str = None) -> OpenAI:
    """Create OpenAI-compatible client for OrcaRouter."""
    return OpenAI(
        base_url=base_url or API_BASE_URL,
        api_key=api_key or API_KEY,
    )


def load_system_prompt() -> str:
    """Load the system prompt for math OCR."""
    return """You are an expert OCR system specialized in reading handwritten mathematics from undergraduate-level answer sheets.

Your task: Convert ONLY the handwritten mathematical content into a complete, compilable LaTeX document.

INCLUDE:
- All handwritten equations and mathematical expressions
- All handwritten text that is part of the solution (like "Solution:", "Let x =", etc.)
- Preserve the spatial layout and logical flow exactly as written

IGNORE (do NOT include these in output):
- Any printed/typed text (headers, footers, instructions)
- Student name, roll number, date, page numbers (e.g. "2025-Shishir KV, VVCHTG 2 of 1")
- Cancelled/crossed-out work
- Rough/scratch work sections
- Any watermarks or stamps

FORMATTING RULES:
1. Output must be a COMPLETE LaTeX document (\\documentclass through \\end{document})
2. Use amsmath, amssymb, amsfonts packages
3. Preserve the exact spatial layout - if content is on the left side, keep it left-aligned
4. Use appropriate environments: align for multi-line equations, equation for single display equations
5. Inline math uses $...$, display math uses \\[...\\] or align/equation environments
6. If something is illegible, mark as \\textit{[illegible]}
7. The output MUST compile without errors using pdflatex
8. The compiled PDF should visually match the layout of the handwritten page

Output ONLY the LaTeX code. No explanations, no markdown, no comments about what you see."""


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_pending_images(images_dir: str, annotations_dir: str) -> list:
    """Get list of images that DON'T have annotations yet."""
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    
    all_images = sorted(images_path.glob("*.png"))
    
    pending = []
    for img in all_images:
        annotation_file = annotations_path / f"{img.stem}.tex"
        if not annotation_file.exists():
            pending.append(img)
    
    return pending


def get_completed_count(annotations_dir: str) -> int:
    """Count how many annotations already exist."""
    annotations_path = Path(annotations_dir)
    if not annotations_path.exists():
        return 0
    return len(list(annotations_path.glob("*.tex")))


def clear_annotations(annotations_dir: str) -> int:
    """Delete all .tex files in annotations directory."""
    annotations_path = Path(annotations_dir)
    if not annotations_path.exists():
        return 0
    
    tex_files = list(annotations_path.glob("*.tex"))
    count = len(tex_files)
    for f in tex_files:
        f.unlink()
    
    return count


def generate_latex_for_image(client: OpenAI, image_path: Path, system_prompt: str) -> str:
    """Send image to vision model and get LaTeX back."""
    img_base64 = encode_image(str(image_path))
    
    suffix = image_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    
    user_message = (
        "OCR this handwritten math page. Convert ONLY the handwritten mathematical "
        "content into LaTeX. Ignore all printed text, student info, page numbers, "
        "cancelled work, and rough work. Preserve the spatial layout exactly. "
        "Output a complete compilable LaTeX document only."
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_message
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.1,
    )
    
    return response.choices[0].message.content


def clean_latex_response(response: str) -> str:
    """Clean the model response to extract pure LaTeX."""
    text = response.strip()
    
    if text.startswith("```latex"):
        text = text[len("```latex"):].strip()
    elif text.startswith("```tex"):
        text = text[len("```tex"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    
    if text.endswith("```"):
        text = text[:-3].strip()
    
    return text


def is_valid_response(latex: str) -> bool:
    """Check if the response is actually valid OCR output (not a refusal)."""
    bad_phrases = [
        "please provide the image",
        "please attach",
        "no handwritten content provided",
        "i cannot see",
        "no image",
        "unable to process",
    ]
    lower = latex.lower()
    for phrase in bad_phrases:
        if phrase in lower:
            return False
    
    # Must have some actual content (more than just boilerplate)
    if len(latex) < 200:
        # Short response - check if it has any math
        if "$" not in latex and "\\[" not in latex and "\\begin{" not in latex:
            return False
    
    return True


def show_status(images_dir: str, annotations_dir: str):
    """Show current progress."""
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    
    total_images = len(list(images_path.glob("*.png")))
    completed = get_completed_count(annotations_dir)
    remaining = total_images - completed

    if total_images == 0:
        print("No images found in", images_dir)
        return

    pct = completed / total_images * 100
    bars = int(completed * 40 / total_images)
    print("=" * 60)
    print("ANNOTATION STATUS")
    print("=" * 60)
    print(f"  Total images:     {total_images}")
    print(f"  Completed:        {completed} ({pct:.1f}%)")
    print(f"  Remaining:        {remaining}")
    print(f"  Progress bar:     [{'#' * bars}{'.' * (40 - bars)}]")
    
    # Load progress file if exists
    progress_file = Path(annotations_dir).parent / "annotation_progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        print(f"\n  Last session:")
        print(f"    Success: {progress.get('success', '?')}")
        print(f"    Failed:  {progress.get('failed', '?')}")
        print(f"    Last image: {progress.get('last_image', '?')}")
    
    if remaining > 0:
        print(f"\n  To continue: python scripts/02_annotate_helper.py --mode auto")
        print(f"  With new key: python scripts/02_annotate_helper.py --mode auto --api-key YOUR_NEW_KEY")


def dry_run(images_dir: str, annotations_dir: str, num_images: int = 5,
            api_key: str = None, base_url: str = None):
    """DRY RUN: Deletes previous annotations, processes N images."""
    print("=" * 60)
    print(f"DRY RUN MODE - Processing {num_images} images")
    print("=" * 60)
    
    # Delete previous dry-run files
    deleted = clear_annotations(annotations_dir)
    if deleted > 0:
        print(f"\n  Cleaned up: Deleted {deleted} previous .tex files")
    
    client = get_client(api_key, base_url)
    system_prompt = load_system_prompt()
    
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    annotations_path.mkdir(parents=True, exist_ok=True)
    
    all_images = sorted(images_path.glob("*.png"))[:num_images]
    
    print(f"\nImages to process: {len(all_images)}")
    print(f"Output dir: {annotations_path}\n")
    
    results = []
    for idx, img_path in enumerate(all_images, 1):
        print(f"[{idx}/{num_images}] Processing: {img_path.name}")
        print(f"  Image size: {img_path.stat().st_size / 1024:.1f} KB")
        
        try:
            start_time = time.time()
            raw_response = generate_latex_for_image(client, img_path, system_prompt)
            latex = clean_latex_response(raw_response)
            elapsed = time.time() - start_time
            
            # Check if valid
            valid = is_valid_response(latex)
            
            # Save annotation
            annotation_file = annotations_path / f"{img_path.stem}.tex"
            annotation_file.write_text(latex, encoding='utf-8')
            
            has_docclass = "\\documentclass" in latex
            has_enddoc = "\\end{document}" in latex
            
            print(f"  Generated in {elapsed:.1f}s ({len(latex)} chars)")
            print(f"  Has documentclass: {'Yes' if has_docclass else 'NO'}")
            print(f"  Has end document: {'Yes' if has_enddoc else 'NO'}")
            print(f"  Valid OCR output: {'Yes' if valid else 'NO - might be refusal'}")
            print(f"  Preview: {latex[:150]}...")
            print()
            
            results.append({
                'image': img_path.name,
                'status': 'success' if valid else 'invalid',
                'chars': len(latex),
                'time': elapsed
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            results.append({
                'image': img_path.name,
                'status': 'failed',
                'error': str(e)
            })
        
        if idx < num_images:
            time.sleep(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("DRY RUN SUMMARY")
    print("=" * 60)
    success = sum(1 for r in results if r['status'] == 'success')
    invalid = sum(1 for r in results if r['status'] == 'invalid')
    failed = sum(1 for r in results if r['status'] == 'failed')
    print(f"  Success: {success}/{num_images}")
    print(f"  Invalid (model refused): {invalid}/{num_images}")
    print(f"  Failed (error): {failed}/{num_images}")
    
    if success > 0:
        avg_time = sum(r.get('time', 0) for r in results if r['status'] == 'success') / success
        avg_chars = sum(r.get('chars', 0) for r in results if r['status'] == 'success') / success
        print(f"  Avg time per image: {avg_time:.1f}s")
        print(f"  Avg LaTeX length: {avg_chars:.0f} chars")
        print(f"  Estimated time for all 6247 images: {avg_time * 6247 / 3600:.1f} hours")
    
    print(f"\n  .tex files in: {annotations_path}")
    print(f"  Next dry-run will DELETE these and regenerate.")
    print(f"  When ready: python scripts/02_annotate_helper.py --mode auto")


def auto_annotate(images_dir: str, annotations_dir: str, 
                  api_key: str = None, base_url: str = None,
                  delay: float = 0.0):
    """
    AUTO MODE: Process all REMAINING unannotated images.
    
    - Automatically resumes from where it left off
    - If credit runs out, just run again with --api-key NEW_KEY
    - Only processes images without existing .tex files
    """
    print("=" * 60)
    print("AUTO ANNOTATION MODE (with resume support)")
    print("=" * 60)
    
    client = get_client(api_key, base_url)
    system_prompt = load_system_prompt()
    
    annotations_path = Path(annotations_dir)
    annotations_path.mkdir(parents=True, exist_ok=True)
    
    # Get only PENDING images (skips already-annotated ones)
    pending = get_pending_images(images_dir, annotations_dir)
    total_images = len(list(Path(images_dir).glob("*.png")))
    completed_before = total_images - len(pending)
    
    print(f"\n  Total images:      {total_images}")
    print(f"  Already completed: {completed_before}")
    print(f"  Remaining:         {len(pending)}")
    print(f"  Delay:             {delay}s between requests (0 = max speed)")
    
    if len(pending) == 0:
        print("\n  All images are already annotated! Nothing to do.")
        print("  Run --mode status to see details.")
        return
    
    print(f"\n  Starting from image: {pending[0].name}")
    print(f"  Press Ctrl+C to stop (progress is saved automatically)\n")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    progress_file = annotations_path.parent / "annotation_progress.json"
    
    try:
        for idx, img_path in enumerate(pending, 1):
            print(f"[{completed_before + idx}/{total_images}] {img_path.name}", end=" ")
            
            try:
                start_time = time.time()
                raw_response = generate_latex_for_image(client, img_path, system_prompt)
                latex = clean_latex_response(raw_response)
                elapsed = time.time() - start_time
                
                # Validate response
                if not is_valid_response(latex):
                    skip_count += 1
                    print(f"SKIPPED (model refused/invalid, {elapsed:.1f}s)")
                    # Don't save invalid responses - will retry next run
                    continue
                
                # Save
                annotation_file = annotations_path / f"{img_path.stem}.tex"
                annotation_file.write_text(latex, encoding='utf-8')
                
                success_count += 1
                print(f"ok ({len(latex)} chars, {elapsed:.1f}s)")
                
            except Exception as e:
                fail_count += 1
                error_str = str(e)
                print(f"FAIL: {error_str[:80]}")
                
                # Rate limit or credit exhausted
                if "429" in error_str or "rate" in error_str.lower():
                    print("\n  Rate limited! Waiting 60s...")
                    time.sleep(60)
                elif "402" in error_str or "insufficient" in error_str.lower() or "credit" in error_str.lower():
                    print("\n" + "=" * 60)
                    print("  CREDIT EXHAUSTED!")
                    print("  To continue with a new API key:")
                    print(f"  python scripts/02_annotate_helper.py --mode auto --api-key YOUR_NEW_KEY")
                    print("=" * 60)
                    break
                elif "401" in error_str or "unauthorized" in error_str.lower():
                    print("\n  Invalid API key! Check your key and try again.")
                    break
            
            # Optional delay between requests (to avoid rate limits)
            if delay > 0:
                time.sleep(delay)
            
            # Save progress every 25 images
            if idx % 25 == 0:
                total_done = completed_before + success_count
                progress = {
                    'total_images': total_images,
                    'completed': total_done,
                    'remaining': total_images - total_done,
                    'this_session_success': success_count,
                    'this_session_failed': fail_count,
                    'this_session_skipped': skip_count,
                    'last_image': img_path.name,
                    'percent': f"{total_done/total_images*100:.1f}%"
                }
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
                pct = total_done / total_images * 100
                print(f"\n  --- Progress: {total_done}/{total_images} ({pct:.1f}%) | Session: +{success_count} ok, {fail_count} fail, {skip_count} skip ---\n")
    
    except KeyboardInterrupt:
        print("\n\n  Stopped by user (Ctrl+C)")
    
    # Final save
    total_done = completed_before + success_count
    progress = {
        'total_images': total_images,
        'completed': total_done,
        'remaining': total_images - total_done,
        'this_session_success': success_count,
        'this_session_failed': fail_count,
        'this_session_skipped': skip_count,
        'last_image': pending[min(idx-1, len(pending)-1)].name if pending else "none",
        'percent': f"{total_done/total_images*100:.1f}%"
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"  This session:")
    print(f"    Success:  {success_count}")
    print(f"    Failed:   {fail_count}")
    print(f"    Skipped:  {skip_count}")
    print(f"\n  Overall progress:")
    print(f"    Completed: {total_done}/{total_images} ({total_done/total_images*100:.1f}%)")
    print(f"    Remaining: {total_images - total_done}")
    
    if total_done < total_images:
        print(f"\n  To continue with new API key:")
        print(f"    python scripts/02_annotate_helper.py --mode auto --api-key YOUR_NEW_KEY")
        print(f"\n  To check status:")
        print(f"    python scripts/02_annotate_helper.py --mode status")


def review_annotations(images_dir: str, annotations_dir: str):
    """REVIEW MODE: Open each annotation + image for manual review."""
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    
    annotations = sorted(annotations_path.glob("*.tex"))
    
    unreviewed = []
    for ann in annotations:
        content = ann.read_text(encoding='utf-8')
        if "% REVIEWED: OK" not in content:
            unreviewed.append(ann)
    
    print(f"Total annotations: {len(annotations)}")
    print(f"Unreviewed: {len(unreviewed)}")
    print("\nControls: ENTER=approve, e=edit, d=delete, s=skip, q=quit\n")
    
    reviewed = 0
    for ann_file in unreviewed:
        img_file = images_path / f"{ann_file.stem}.png"
        if not img_file.exists():
            continue
        
        content = ann_file.read_text(encoding='utf-8')
        
        print(f"\n{'='*60}")
        print(f"Image: {img_file.name}")
        print(f"Annotation: {ann_file.name} ({len(content)} chars)")
        print(f"{'='*60}")
        print(content[:400] + ("..." if len(content) > 400 else ""))
        
        os.startfile(str(img_file))
        
        choice = input("\n[ENTER=approve, e=edit, d=delete, s=skip, q=quit]: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'e':
            os.startfile(str(ann_file))
            input("Press ENTER when done editing...")
        elif choice == 'd':
            ann_file.unlink()
            print("  Deleted.")
            continue
        elif choice == 's':
            continue
        
        with open(ann_file, 'a', encoding='utf-8') as f:
            f.write("\n% REVIEWED: OK\n")
        reviewed += 1
        print(f"  Approved ({reviewed} total)")
    
    print(f"\nReview session done. Approved: {reviewed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LaTeX annotations using vision models via OrcaRouter"
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "auto", "resume", "review", "status"],
        default="dry-run",
        help="Mode: dry-run (test N images), auto/resume (all remaining), review, status"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=r"d:\Kiro\dataset\images",
        help="Directory containing prepared images"
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default=r"d:\Kiro\dataset\annotations",
        help="Directory to save LaTeX annotations"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images for dry-run mode (default: 5)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (use to update key when resuming with new credit)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (default: https://api.orcarouter.ai/v1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between API calls in seconds (default: 0.0, no delay)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "status":
        show_status(args.images_dir, args.annotations_dir)
    
    elif args.mode == "dry-run":
        dry_run(
            args.images_dir, args.annotations_dir, 
            num_images=args.num_images,
            api_key=args.api_key, base_url=args.base_url
        )
    
    elif args.mode in ("auto", "resume"):
        auto_annotate(
            args.images_dir, args.annotations_dir,
            api_key=args.api_key, base_url=args.base_url,
            delay=args.delay
        )
    
    elif args.mode == "review":
        review_annotations(args.images_dir, args.annotations_dir)
