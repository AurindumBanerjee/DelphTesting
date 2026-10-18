@echo off
REM Same as run_benchmark.bat, but verbose and fully logged to bench.log
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo === start %date% %time% > bench.log
if not exist .venv\Scripts\python.exe python -m venv .venv >> bench.log 2>&1
.venv\Scripts\python.exe -m pip install --progress-bar off -r requirements.txt >> bench.log 2>&1
if errorlevel 1 (echo === PIP FAILED >> bench.log & exit /b 1)
echo === deps installed %time% >> bench.log
if not exist eval\corpus mkdir eval\corpus
copy /Y books\*.pdf eval\corpus\ >> bench.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --methods recursive --sizes 500 1000 2000 --k 1 3 5 10 >> bench.log 2>&1
if errorlevel 1 (echo === EVAL FAILED >> bench.log & exit /b 1)
echo === DONE %date% %time% >> bench.log
