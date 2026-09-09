<#
  OPT-IN. Not run automatically by anything in this folder.

  Registers a Windows Task Scheduler entry that relaunches
  ".\run_v3.ps1 train" at logon, so if your PC crashes and REBOOTS (not just
  the python process dying -- run_v3.ps1's own retry loop already handles
  that), training auto-resumes from the last checkpoint the next time you (or
  Windows, if you also enable auto-login) log in, with no manual step.

  This is the one piece of the power-fail story that has to live outside
  D:\Claude Code\BaiduOCR-v3, because a scheduled task is Windows system
  configuration, not a project file. Everything it points at still lives
  entirely inside this folder.

  Run once, manually, when you want this:
      .\register_startup_resume.ps1

  Remove it any time with:
      Unregister-ScheduledTask -TaskName 'BaiduOCR-v3-AutoResume' -Confirm:$false
#>
param([string]$TaskName = 'BaiduOCR-v3-AutoResume')

$V3 = Split-Path -Parent $MyInvocation.MyCommand.Path
$runScript = Join-Path $V3 'run_v3.ps1'
if (-not (Test-Path $runScript)) { throw "run_v3.ps1 not found next to this script" }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`" train" `
  -WorkingDirectory $V3

$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) `
  -ExecutionTimeLimit (New-TimeSpan -Days 3)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
  -Description 'Auto-resumes BaiduOCR-v3 training after a reboot. See D:\Claude Code\BaiduOCR-v3\register_startup_resume.ps1' `
  -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName' - training will relaunch at your next logon." -ForegroundColor Green
Write-Host "Remove it with: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor DarkGray
