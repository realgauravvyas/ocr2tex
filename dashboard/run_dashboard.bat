@echo off
REM Start the OCR2TeX dashboard in its own window, detached from this console.
REM It keeps running even after you close THIS window or your browser tab.
REM Stop it with the "Stop" button in the dashboard header, or close its window.
start "OCR2TeX Dashboard" /min python "D:\ocr2tex\dashboard\app.py"
echo Dashboard starting on http://127.0.0.1:5001  (running in a minimized window)
echo Closing this window will NOT stop it. Use the Stop button in the dashboard to kill it.
timeout /t 3 >nul
