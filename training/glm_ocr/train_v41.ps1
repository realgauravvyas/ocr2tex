# Launch (or resume) v4.1 training. Run this when you go to sleep.
# - Low process priority so the desktop stays responsive
# - Checkpoints every 150 steps (a crash/shutdown loses at most ~150 steps)
# - Auto-resumes from the latest checkpoint if one exists; else warm-starts from v4
# Optional: pass -Throttle 0.3 to keep the PC usable while training (slows it down).
param([double]$Throttle = 0.0, [int]$Workers = 2)

$out   = "D:\ocr2tex\output\glm-ocr-math-v4.1"
$train = "D:\ocr2tex\dashboard\train_glm_ocr.py"

$argList = @(
  "-u", $train,
  "--output-dir", $out,
  "--warm-start-adapter", "D:\ocr2tex\output\glm-ocr-math-v4\final",
  "--train-file", "D:\ocr2tex\workspace\9_split_v4.1\train.jsonl",
  "--val-file",   "D:\ocr2tex\workspace\9_split_v4.1\val.jsonl",
  "--image-base", "D:\ocr2tex\workspace\9_split",
  "--epochs", "2", "--lr", "1e-5", "--batch-size", "1", "--grad-accum", "8",
  "--max-length", "3584", "--max-image-tokens", "1536",
  "--eval-samples", "200", "--early-stopping-patience", "5",
  "--workers", "$Workers", "--seed", "42", "--tf32",
  "--save-steps", "150"
)
if ($Throttle -gt 0) { $argList += @("--throttle", "$Throttle") }

# resume if a checkpoint exists
$ckpts = Get-ChildItem -Path $out -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue
if ($ckpts) {
  $argList += "--resume"
  Write-Host "Found checkpoint(s) - RESUMING from the latest." -ForegroundColor Green
} else {
  Write-Host "No checkpoint - starting fresh (warm-start from v4)." -ForegroundColor Yellow
}

New-Item -ItemType Directory -Force $out | Out-Null
# BelowNormal priority keeps Windows responsive for browsing
$p = Start-Process python -ArgumentList $argList -PassThru -WindowStyle Hidden `
       -RedirectStandardOutput "$out\train_console.log" -RedirectStandardError "$out\train_err.log"
Start-Sleep -Seconds 2
try { $p.PriorityClass = "BelowNormal" } catch {}
Write-Host "v4.1 training started (pid $($p.Id)). Watch it in the dashboard Fine-tuning tab." -ForegroundColor Cyan
Write-Host "Checkpoints every 150 steps -> safe to shut down; just run this script again to resume."
