<#
  BaiduOCR v3 driver.

  Everything runs out of this folder. The only things it reads outside it are
  read-only: the 9_split dataset, the v1 adapter (probe stage only), the existing
  benchmark_results.json (printed for comparison) and the BaiduOCR venv
  interpreter. Nothing outside D:\Claude Code\BaiduOCR-v3 is written.

  POWER-FAIL SAFETY
  ------------------
  train_baidu_v3.py checkpoints to two alternating slots and only swings an
  atomically-replaced pointer file over once a slot is fully written, so a
  power cut mid-save can never corrupt the checkpoint resume reads (see the
  comment block above save_checkpoint() in train_baidu_v3.py). It also
  auto-resumes from that checkpoint by default -- no flag needed, rerunning
  the exact same command after a crash just continues.

  'train'/'resume' in THIS script additionally wrap the python process in a
  retry loop: if it dies for any reason (driver crash, OOM cascade, the
  process getting killed), it is relaunched after a short delay and picks up
  from the last completed checkpoint automatically. That covers a crash that
  does NOT reboot the machine.

  It can NOT survive the machine actually rebooting on its own, because a
  PowerShell loop dies with the session. For that, see
  register_startup_resume.ps1 in this folder (opt-in, registers a Windows
  Task Scheduler entry to relaunch training at logon) -- it is not run
  automatically; run it yourself if you want reboot-survival too.

  Usage:
    .\run_v3.ps1 smoke        # ~5 min   sanity-check the pipeline end to end
    .\run_v3.ps1 probe        # ~25 min  A/B the decode fixes on the v1 adapter
    .\run_v3.ps1 train        # ~35-45 h 2 epochs with the corrected LoRA, auto-retries on crash
    .\run_v3.ps1 resume       #          same as 'train' -- resume is automatic; kept as an alias
    .\run_v3.ps1 gen          # ~4-6 h   700-page test generation (already resumable per-page)
    .\run_v3.ps1 score        # ~10 min  metrics + comparison table
    .\run_v3.ps1 gen-ring-ab  #          same adapter, ring ON, for the ablation
#>
param(
  [Parameter(Position = 0)][ValidateSet('smoke','probe','train','resume','gen','score','gen-ring-ab')]
  [string]$Stage = 'smoke',
  [int]$MaxRetries = 500,     # generous: at ~40h and one crash every few hours this still won't run out
  [int]$RetryDelaySec = 30
)

# NOT 'Stop': every python subprocess below is run with 2>&1 (to Tee-Object
# logging to file), and PowerShell 5.1 wraps each stderr line from a native
# process into a NativeCommandError when redirected this way. Under 'Stop'
# preference that promotes to a script-terminating exception on the FIRST
# warning the model prints during load (there are several), killing the
# whole retry loop in seconds instead of training for hours. Our own
# explicit `throw` below still terminates regardless of this setting.
$ErrorActionPreference = 'Continue'
$V3 = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY = 'D:\Claude Code\BaiduOCR\venv\Scripts\python.exe'   # used, never modified
if (-not (Test-Path $PY)) { throw "python not found at $PY - pass your own interpreter" }

$logs = Join-Path $V3 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $logs "$Stage`_$stamp.log"
Write-Host "stage=$Stage  log=$log" -ForegroundColor Cyan

function Invoke-Resilient {
  param([string[]]$PyArgs)
  $attempt = 0
  while ($true) {
    $attempt++
    $tag = "[attempt $attempt/$MaxRetries $(Get-Date -Format 'HH:mm:ss')]"
    Write-Host "$tag launching: python $($PyArgs -join ' ')" -ForegroundColor Yellow
    Add-Content -Path $log -Value "`n===== $tag launching =====`n"
    & $PY @PyArgs 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    if ($code -eq 0) {
      Write-Host "[$(Get-Date -Format 'HH:mm:ss')] process exited cleanly (0) - done" -ForegroundColor Green
      return
    }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] process exited with code $code - will relaunch and " `
               "auto-resume from the last checkpoint" -ForegroundColor Red
    Add-Content -Path $log -Value "`n===== exited $code, retrying in ${RetryDelaySec}s =====`n"
    if ($attempt -ge $MaxRetries) {
      Write-Host "giving up after $MaxRetries attempts - see $log" -ForegroundColor Red
      return
    }
    Start-Sleep -Seconds $RetryDelaySec
  }
}

$trainArgs = @(
  (Join-Path $V3 'train_baidu_v3.py'),
  '--epochs', '2', '--lora-r', '32', '--lora-alpha', '64', '--expert-rank', '8',
  '--metric-eval-steps', '400', '--metric-eval-pages', '24',
  '--eval-steps', '400', '--save-steps', '100', '--save-every-s', '300'
)

switch ($Stage) {
  'smoke' {
    & $PY (Join-Path $V3 'train_baidu_v3.py') --smoke 2>&1 | Tee-Object -FilePath $log
  }
  'probe' {
    & $PY (Join-Path $V3 'probe_decode.py') --pages 24 2>&1 | Tee-Object -FilePath $log
  }
  'train'  { Invoke-Resilient -PyArgs $trainArgs }
  'resume' { Invoke-Resilient -PyArgs $trainArgs }   # auto-resume is unconditional; alias for clarity
  'gen' {
    # per-page resumable already (skips pages with a completed .raw/.tex/.time);
    # still wrapped so a crash mid-run doesn't require you to notice and relaunch it
    Invoke-Resilient -PyArgs @((Join-Path $V3 'gen_baidu_v3.py'), '--label', 'Baidu_OCR_FT_v3')
  }
  'gen-ring-ab' {
    Invoke-Resilient -PyArgs @((Join-Path $V3 'gen_baidu_v3.py'), '--label', 'Baidu_OCR_FT_v3_ring', '--ring')
  }
  'score' {
    & $PY (Join-Path $V3 'score_v3.py') --label Baidu_OCR_FT_v3 2>&1 | Tee-Object -FilePath $log
  }
}
